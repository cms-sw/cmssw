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
  refHitLabel_ = pset.getParameter<edm::InputTag>("refHit");
  recHitLabel_ = pset.getParameter<edm::InputTag>("recHit");

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
  int nRefHitBarrel = 0, nRefHitEndcap = 0;
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
      h_.refHitOccupancyBarrel_wheel->Fill(ring);
      h_.refHitOccupancyBarrel_station->Fill(station);
      h_.refHitOccupancyBarrel_wheel_station->Fill(ring, station);
    }
    else
    {
      h_.refHitOccupancyEndcap_disk->Fill(region*station);
      h_.refHitOccupancyEndcap_disk_ring->Fill(region*station, ring);
    }

  }
  h_.nRefHitBarrel->Fill(nRefHitBarrel);
  h_.nRefHitEndcap->Fill(nRefHitEndcap);

  // Loop over recHits, fill histograms which does not need associations
  int sumClusterSizeBarrel = 0, sumClusterSizeEndcap = 0;
  int nRecHitBarrel = 0, nRecHitEndcap = 0;
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
      ++nRecHitBarrel;
      sumClusterSizeBarrel += recHitIter->clusterSize();
      h_.clusterSizeBarrel->Fill(recHitIter->clusterSize());
      h_.recHitOccupancyBarrel_wheel->Fill(ring);
      h_.recHitOccupancyBarrel_station->Fill(station);
      h_.recHitOccupancyBarrel_wheel_station->Fill(ring, station);
    }
    else
    {
      ++nRecHitEndcap;
      sumClusterSizeEndcap += recHitIter->clusterSize();
      h_.clusterSizeEndcap->Fill(recHitIter->clusterSize());
      h_.recHitOccupancyEndcap_disk->Fill(region*station);
      h_.recHitOccupancyEndcap_disk_ring->Fill(ring, region*station);
    }

  }
  const double nRecHit = nRecHitBarrel+nRecHitEndcap;
  h_.nRecHitBarrel->Fill(nRecHitBarrel);
  h_.nRecHitEndcap->Fill(nRecHitEndcap);
  if ( nRecHit > 0 )
  {
    const int sumClusterSize = sumClusterSizeBarrel+sumClusterSizeEndcap;
    h_.avgClusterSize->Fill(double(sumClusterSize)/nRecHit);

    if ( nRecHitBarrel > 0 )
    {
      h_.avgClusterSizeBarrel->Fill(double(sumClusterSizeBarrel)/nRecHitBarrel);
    }
    if ( nRecHitEndcap > 0 )
    {
      h_.avgClusterSizeEndcap->Fill(double(sumClusterSizeEndcap)/nRecHitEndcap);
    }
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
    const double errX = sqrt(recHitIter->localPositionError().xx());
    const double dX = recX - refX;
    const double pull = errX == 0 ? -999 : dX/errX;

    //const GlobalPoint refPos = roll->toGlobal(refHitIter->localPosition());
    //const GlobalPoint recPos = roll->toGlobal(recHitIter->localPosition());

    if ( region == 0 )
    {
      h_.resBarrel->Fill(dX);
      h_.pullBarrel->Fill(pull);
      h_.matchOccupancyBarrel_wheel->Fill(ring);
      h_.matchOccupancyBarrel_station->Fill(station);
      h_.matchOccupancyBarrel_wheel_station->Fill(ring, station);

      h_.res_wheel_res->Fill(ring, dX);
      h_.res_station_res->Fill(station, dX);
      h_.pull_wheel_pull->Fill(ring, pull);
      h_.pull_station_pull->Fill(station, pull);
    }
    else
    {
      h_.resEndcap->Fill(dX);
      h_.pullEndcap->Fill(pull);
      h_.matchOccupancyEndcap_disk->Fill(region*station);
      h_.matchOccupancyEndcap_disk_ring->Fill(region*station, ring);

      h_.res_disk_res->Fill(region*station, dX);
      h_.res_ring_res->Fill(ring, dX);
      h_.pull_disk_pull->Fill(region*station, pull);
      h_.pull_ring_pull->Fill(ring, pull);
    }
  }

/*
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
        h_.nUrefHitOccupancyBarrel_wheel->Fill(ring);
        h_.nUrefHitOccupancyBarrel_wheel_ring->Fill(ring, station);
      }
      else
      {
        h_.nUnMatchedRefHit_disk->Fill(region*station);
        h_.nUnMatchedRefHit_disk_ring->Fill(region*station, ring);
      }
    }
  }
*/

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
        h_.umOccupancyBarrel_wheel->Fill(ring);
        h_.umOccupancyBarrel_station->Fill(station);
        h_.umOccupancyBarrel_wheel_station->Fill(ring, station);
      }
      else
      {
        h_.umOccupancyEndcap_disk->Fill(region*station);
        h_.umOccupancyEndcap_disk_ring->Fill(region*station, ring);
      }

      //const GlobalPoint pos = roll->toGlobal(recHitIter->localPosition());
      //h_[HName::NoisyHitEta]->Fill(pos.eta());
    }
  }
}

DEFINE_FWK_MODULE(RPCPointVsRecHit);

