#include "Validation/RPCRecHits/interface/RPCPointVsRecHit.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace std;

typedef MonitorElement* MEP;

RPCPointVsRecHit::RPCPointVsRecHit(const edm::ParameterSet& pset)
{
  rootFileName_ = pset.getUntrackedParameter<string>("rootFileName", "");
  refHitLabel_ = pset.getParameter<edm::InputTag>("refHit");
  recHitLabel_ = pset.getParameter<edm::InputTag>("recHit");

  isStandAloneMode_ = pset.getUntrackedParameter<bool>("standAloneMode", false);

  dbe_ = edm::Service<DQMStore>().operator->();
  if ( !dbe_ )
  {
    edm::LogError("RPCPointVsRecHit") << "No DQMStore instance\n";
    return;
  }

  // Book MonitorElements
  const std::string subDir = pset.getParameter<std::string>("subDir");
  h_.bookHistograms(dbe_, subDir);
}

RPCPointVsRecHit::~RPCPointVsRecHit()
{
  if ( dbe_ )
  {
    if ( !rootFileName_.empty() ) dbe_->save(rootFileName_);
  }
}

void RPCPointVsRecHit::beginJob()
{
}

void RPCPointVsRecHit::endJob()
{
}

void RPCPointVsRecHit::analyze(const edm::Event& event, const edm::EventSetup& eventSetup)
{
  if ( !dbe_ )
  {
    edm::LogError("RPCPointVsRecHit") << "No DQMStore instance\n";
    return;
  }

  // Get the RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);

  // Retrieve RefHits from the event
  edm::Handle<RPCRecHitCollection> refHitHandle;
  if ( !event.getByLabel(refHitLabel_, refHitHandle) )
  {
    edm::LogInfo("RPCPointVsRecHit") << "Cannot find reference hit collection\n";
    return;
  }

  // Retrieve RecHits from the event
  edm::Handle<RPCRecHitCollection> recHitHandle;
  if ( !event.getByLabel(recHitLabel_, recHitHandle) )
  {
    edm::LogInfo("RPCPointVsRecHit") << "Cannot find recHit collection\n";
    return;
  }

  typedef RPCRecHitCollection::const_iterator RecHitIter;

  // Loop over refHits, fill histograms which does not need associations
  for ( RecHitIter refHitIter = refHitHandle->begin();
        refHitIter != refHitHandle->end(); ++refHitIter )
  {
    const RPCDetId detId = static_cast<const RPCDetId>(refHitIter->rpcId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId()));
    if ( !roll ) continue;

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subSector = roll->id().subsector();

    if ( region == 0 )
    {
      h_.nRefHit_W->Fill(ring);
      h_.nRefHit_WvsR->Fill(ring, station);
    }
    else
    {
      h_.nRefHit_D->Fill(region*station);
      h_.nRefHit_DvsR->Fill(region*station, ring);
    }

    const GlobalPoint pos = roll->toGlobal(refHitIter->localPosition());
    //h_[HName::RefHitEta]->Fill(pos.eta());
  }

  // Loop over recHits, fill histograms which does not need associations
  for ( RecHitIter recHitIter = recHitHandle->begin();
        recHitIter != recHitHandle->end(); ++recHitIter )
  {
    const RPCDetId detId = static_cast<const RPCDetId>(recHitIter->rpcId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId()));
    if ( !roll ) continue;

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = (roll->id().station());
    //const int layer = roll->id().layer();
    //const int subSector = roll->id().subsector();

    h_.clusterSize->Fill(recHitIter->clusterSize());

    if ( region == 0 )
    {
      h_.nRecHit_W->Fill(ring);
      h_.nRecHit_WvsR->Fill(ring, station);
    }
    else
    {
      h_.nRecHit_D->Fill(region*station);
      h_.nRecHit_DvsR->Fill(ring, region*station);
    }


    const GlobalPoint pos = roll->toGlobal(recHitIter->localPosition());
    //h_[HName::RecHitEta]->Fill(pos.eta());
  }

  // Start matching RefHits to RecHits
  typedef std::map<RecHitIter, RecHitIter> RecToRecHitMap;
  RecToRecHitMap refToRecHitMap;

  for ( RecHitIter refHitIter = refHitHandle->begin();
        refHitIter != refHitHandle->end(); ++refHitIter )
  {
    const RPCDetId refDetId = static_cast<const RPCDetId>(refHitIter->rpcId());
    const RPCRoll* refRoll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(refDetId));
    if ( !refRoll ) continue;

    const double refX = refHitIter->localPosition().x();

    for ( RecHitIter recHitIter = recHitHandle->begin();
          recHitIter != recHitHandle->end(); ++recHitIter )
    {
      const RPCDetId recDetId = static_cast<const RPCDetId>(recHitIter->rpcId());
      const RPCRoll* recRoll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(recDetId));
      if ( !recRoll ) continue;

      if ( refDetId != recDetId ) continue;

      const double recX = recHitIter->localPosition().x();
      const double newDx = fabs(recX - refX);

      // Associate RefHit to RecHit
      RecToRecHitMap::const_iterator prevRefToReco = refToRecHitMap.find(refHitIter);
      if ( prevRefToReco == refToRecHitMap.end() )
      {
        refToRecHitMap.insert(std::make_pair(refHitIter, recHitIter));
      }
      else
      {
        const double oldDx = fabs(prevRefToReco->second->localPosition().x() - refX);

        if ( newDx < oldDx )
        {
          refToRecHitMap[refHitIter] = recHitIter;
        }
      }
    }
  }

  // Now we have refHit-recHit mapping
  // So we can fill up relavant histograms
  for ( RecToRecHitMap::const_iterator match = refToRecHitMap.begin();
        match != refToRecHitMap.end(); ++match )
  {
    RecHitIter refHitIter = match->first;
    RecHitIter recHitIter = match->second;

    const RPCDetId detId = static_cast<const RPCDetId>(refHitIter->rpcId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId));

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subsector = roll->id().subsector();

    const double refX = refHitIter->localPosition().x();
    const double recX = recHitIter->localPosition().x();
    const double errX = recHitIter->localPositionError().xx();
    const double dX = recX - refX;
    const double pull = errX == 0 ? -999 : dX/errX;

    const GlobalPoint refPos = roll->toGlobal(refHitIter->localPosition());
    const GlobalPoint recPos = roll->toGlobal(recHitIter->localPosition());

    if ( region == 0 )
    {
      h_.res_W->Fill(dX);
      h_.pull_W->Fill(pull);
      h_.nMatchedRefHit_W->Fill(ring);
      h_.nMatchedRefHit_WvsR->Fill(ring, station);

      h_.res2_W->Fill(ring, dX);
      h_.pull2_W->Fill(ring, pull);
    }
    else
    {
      h_.res_D->Fill(dX);
      h_.pull_D->Fill(pull);
      h_.nMatchedRefHit_D->Fill(region*station);
      h_.nMatchedRefHit_DvsR->Fill(region*station, ring);

      h_.res2_D->Fill(region*station, dX);
      h_.pull2_D->Fill(region*station, pull);
    }
  }

  // Find Lost hits
  for ( RecHitIter refHitIter = refHitHandle->begin();
        refHitIter != refHitHandle->end(); ++refHitIter )
  {
    const RPCDetId detId = static_cast<const RPCDetId>(refHitIter->rpcId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId));

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subsector = roll->id().subsector();

    bool matched = false;
    for ( RecToRecHitMap::const_iterator match = refToRecHitMap.begin();
          match != refToRecHitMap.end(); ++match )
    {
      if ( refHitIter == match->first )
      {
        matched = true;
        break;
      }
    }

    if ( !matched )
    {
      if ( region == 0 )
      {
        h_.nUnMatchedRefHit_W->Fill(ring);
        h_.nUnMatchedRefHit_WvsR->Fill(ring, station);
      }
      else
      {
        h_.nUnMatchedRefHit_D->Fill(region*station);
        h_.nUnMatchedRefHit_DvsR->Fill(region*station, ring);
      }
    }
  }

  // Find Noisy hits
  for ( RecHitIter recHitIter = recHitHandle->begin();
        recHitIter != recHitHandle->end(); ++recHitIter )
  {
    const RPCDetId detId = static_cast<const RPCDetId>(recHitIter->rpcId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId));

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subsector = roll->id().subsector();

    bool matched = false;
    for ( RecToRecHitMap::const_iterator match = refToRecHitMap.begin();
          match != refToRecHitMap.end(); ++match )
    {
      if ( recHitIter == match->second )
      {
        matched = true;
        break;
      }
    }

    if ( !matched )
    {
      if ( region == 0 )
      {
        h_.nUnMatchedRecHit_W->Fill(ring);
        h_.nUnMatchedRecHit_WvsR->Fill(ring, station);
      }
      else
      {
        h_.nUnMatchedRecHit_D->Fill(region*station);
        h_.nUnMatchedRecHit_DvsR->Fill(region*station, ring);
      }

      const GlobalPoint pos = roll->toGlobal(recHitIter->localPosition());
      //h_[HName::NoisyHitEta]->Fill(pos.eta());
    }
  }
}

DEFINE_FWK_MODULE(RPCPointVsRecHit);

