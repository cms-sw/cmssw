#include "Validation/RPCRecHits/interface/RPCPointVsRecHit.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace std;

struct HName
{
  enum
  {
    ClusterSize, Res, Pull,

    NRefHit_Wheel, NRefHit_Disk,
    NRecHit_Wheel, NRecHit_Disk,

    NLostHit_Wheel, NLostHit_Disk,
    NNoisyHit_Wheel, NNoisyHit_Disk,

    NMatchedRefHit_Wheel, NMatchedRefHit_Disk,
    NMatchedRecHit_Wheel, NMatchedRecHit_Disk,

    RefHitEta, RecHitEta, MatchedRecHitEta, NoisyHitEta,

    NRefHitRZ,
    NMatchedRefHitRZ, NMatchedRecHitRZ,

    NRefHitXY_WM2, NRefHitXY_WM1, NRefHitXY_W00, NRefHitXY_WP1, NRefHitXY_WP2,
    NRefHitXY_DM3, NRefHitXY_DM2, NRefHitXY_DM1, NRefHitXY_DP1, NRefHitXY_DP2, NRefHitXY_DP3,

    NMatchedRefHitXY_WM2, NMatchedRefHitXY_WM1, NMatchedRefHitXY_W00, NMatchedRefHitXY_WP1, NMatchedRefHitXY_WP2,
    NMatchedRecHitXY_WM2, NMatchedRecHitXY_WM1, NMatchedRecHitXY_W00, NMatchedRecHitXY_WP1, NMatchedRecHitXY_WP2,
    NMatchedRefHitXY_DM3, NMatchedRefHitXY_DM2, NMatchedRefHitXY_DM1, NMatchedRefHitXY_DP1, NMatchedRefHitXY_DP2, NMatchedRefHitXY_DP3,
    NMatchedRecHitXY_DM3, NMatchedRecHitXY_DM2, NMatchedRecHitXY_DM1, NMatchedRecHitXY_DP1, NMatchedRecHitXY_DP2, NMatchedRecHitXY_DP3,

    Res_WM2, Res_WM1, Res_W00, Res_WP1, Res_WP2,
    Res_DM3, Res_DM2, Res_DM1, Res_DP1, Res_DP2, Res_DP3,

    Pull_WM2, Pull_WM1, Pull_W00, Pull_WP1, Pull_WP2,
    Pull_DM3, Pull_DM2, Pull_DM1, Pull_DP1, Pull_DP2, Pull_DP3,

    END
  };
};

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
  std::string subDir = pset.getParameter<std::string>("subDir");
  dbe_->setCurrentFolder(subDir);

  // Global plots
  h_[HName::ClusterSize] = dbe_->book1D("ClusterSize", "Cluster size;Cluster size", 11, -0.5, 10.5);

  h_[HName::Res] = dbe_->book1D("Res", "Global residuals;Residual [cm]", 100, -8, 8);
  h_[HName::Pull] = dbe_->book1D("Pull", "Global pulls;Pull", 100, -5, 5);

  h_[HName::NRefHit_Wheel] = dbe_->book1D("NRefHit_Wheel", "Number of reference Hits;Wheel", 5, -2.5, 2.5);
  h_[HName::NRefHit_Disk] = dbe_->book1D("NRefHit_Disk", "Number of reference Hits;Disk", 7, -3.5, 3.5);

  h_[HName::NRecHit_Wheel] = dbe_->book1D("NRecHit_Wheel", "Number of RecHits;Wheel", 5, -2.5, 2.5);
  h_[HName::NRecHit_Disk] = dbe_->book1D("NRecHit_Disk", "Number of RecHits;Disk", 7, -3.5, 3.5);

  h_[HName::NLostHit_Wheel] = dbe_->book1D("NLostHit_Wheel", "Number of lost hits;Wheel", 5, -2.5, 2.5);
  h_[HName::NLostHit_Disk] = dbe_->book1D("NLostHit_Disk", "Number of lost hits;Disk", 7, -3.5, 3.5);

  h_[HName::NNoisyHit_Wheel] = dbe_->book1D("NNoisyHit_Wheel", "Number of noisy hits;Wheel", 5, -2.5, 2.5);
  h_[HName::NNoisyHit_Disk] = dbe_->book1D("NNoisyHit_Disk", "Number of noisy hits;Disk", 7, -3.5, 3.5);

  h_[HName::NMatchedRefHit_Wheel] = dbe_->book1D("NMatchedRefHit_Wheel", "Number of Matched reference Hits;Wheel", 5, -2.5, 2.5);
  h_[HName::NMatchedRefHit_Disk] = dbe_->book1D("NMatchedRefHit_Disk", "Number of Matched reference Hits;Disk", 7, -3.5, 3.5);

  h_[HName::NMatchedRecHit_Wheel] = dbe_->book1D("NMatchedRecHit_Wheel", "Number of Matched RecHits;Wheel", 5, -2.5, 2.5);
  h_[HName::NMatchedRecHit_Disk] = dbe_->book1D("NMatchedRecHit_Disk", "Number of Matched RecHits;Disk", 7, -3.5, 3.5);

  h_[HName::RefHitEta] = dbe_->book1D("RefHitEta", "Number of reference Hits vs #eta;Pseudorapidity #eta", 100, -2.5, 2.5);
  h_[HName::RecHitEta] = dbe_->book1D("RecHitEta", "Number of recHits vs #eta;Pseudorapidity #eta", 100, -2.5, 2.5);
  h_[HName::NoisyHitEta] = dbe_->book1D("NoisyHitEta", "Number of noisy recHits vs #eta;Pseudorapidity #eta", 100, -2.5, 2.5);
  h_[HName::MatchedRecHitEta] = dbe_->book1D("MatchedRecHitEta", "Number of matched recHits vs Eta;Pseudorapidity #eta", 100, -2.5, 2.5);

  // XY overview
  if ( isStandAloneMode_ )
  {
    const int nBin = 1000;
    const double xmin = -1000, xmax = 1000;
    const double ymin = -1000, ymax = 1000;

    h_[HName::NRefHitRZ] = dbe_->book2D("NRefHitRZ", "Number of RefHits;Z;R", nBin, -1100, 1100, nBin, 0, xmax);

    h_[HName::NMatchedRefHitRZ] = dbe_->book2D("NMatchedRefHitRZ", "Number of Matched RefHits;Z;R", nBin, -1100, 1100, nBin, 0, xmax);
    h_[HName::NMatchedRecHitRZ] = dbe_->book2D("NMatchedRecHitRZ", "Number of Matched RecHits;Z;R", nBin, -1100, 1100, nBin, 0, xmax);

    h_[HName::NRefHitXY_WM2] = dbe_->book2D("NRefHitXY_WM2", "Number of RefHits Wheel -2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_WM1] = dbe_->book2D("NRefHitXY_WM1", "Number of RefHits Wheel -1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_W00] = dbe_->book2D("NRefHitXY_W00", "Number of RefHits Wheel 0;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_WP1] = dbe_->book2D("NRefHitXY_WP1", "Number of RefHits Wheel +1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_WP2] = dbe_->book2D("NRefHitXY_WP2", "Number of RefHits Wheel +2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);

    h_[HName::NRefHitXY_DM3] = dbe_->book2D("NRefHitXY_DM3", "Number of RefHits Disk -3;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_DM2] = dbe_->book2D("NRefHitXY_DM2", "Number of RefHits Disk -2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_DM1] = dbe_->book2D("NRefHitXY_DM1", "Number of RefHits Disk -1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_DP1] = dbe_->book2D("NRefHitXY_DP1", "Number of RefHits Disk +1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_DP2] = dbe_->book2D("NRefHitXY_DP2", "Number of RefHits Disk +2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NRefHitXY_DP3] = dbe_->book2D("NRefHitXY_DP3", "Number of RefHits Disk +3;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);

    h_[HName::NMatchedRefHitXY_WM2] = dbe_->book2D("NMatchedRefHitXY_WM2", "Number of Matched RefHits Wheel -2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_WM1] = dbe_->book2D("NMatchedRefHitXY_WM1", "Number of Matched RefHits Wheel -1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_W00] = dbe_->book2D("NMatchedRefHitXY_W00", "Number of Matched RefHits Wheel 0;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_WP1] = dbe_->book2D("NMatchedRefHitXY_WP1", "Number of Matched RefHits Wheel +1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_WP2] = dbe_->book2D("NMatchedRefHitXY_WP2", "Number of Matched RefHits Wheel +2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);

    h_[HName::NMatchedRefHitXY_DM3] = dbe_->book2D("NMatchedRefHitXY_DM3", "Number of Matched RefHits Disk -3;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_DM2] = dbe_->book2D("NMatchedRefHitXY_DM2", "Number of Matched RefHits Disk -2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_DM1] = dbe_->book2D("NMatchedRefHitXY_DM1", "Number of Matched RefHits Disk -1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_DP1] = dbe_->book2D("NMatchedRefHitXY_DP1", "Number of Matched RefHits Disk +1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_DP2] = dbe_->book2D("NMatchedRefHitXY_DP2", "Number of Matched RefHits Disk +2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRefHitXY_DP3] = dbe_->book2D("NMatchedRefHitXY_DP3", "Number of Matched RefHits Disk +3;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);

    h_[HName::NMatchedRecHitXY_WM2] = dbe_->book2D("NMatchedRecHitXY_WM2", "Number of Matched RecHits Wheel -2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_WM1] = dbe_->book2D("NMatchedRecHitXY_WM1", "Number of Matched RecHits Wheel -1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_W00] = dbe_->book2D("NMatchedRecHitXY_W00", "Number of Matched RecHits Wheel 0;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_WP1] = dbe_->book2D("NMatchedRecHitXY_WP1", "Number of Matched RecHits Wheel +1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_WP2] = dbe_->book2D("NMatchedRecHitXY_WP2", "Number of Matched RecHits Wheel +2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);

    h_[HName::NMatchedRecHitXY_DM3] = dbe_->book2D("NMatchedRecHitXY_DM3", "Number of Matched RecHits Disk -3;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_DM2] = dbe_->book2D("NMatchedRecHitXY_DM2", "Number of Matched RecHits Disk -2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_DM1] = dbe_->book2D("NMatchedRecHitXY_DM1", "Number of Matched RecHits Disk -1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_DP1] = dbe_->book2D("NMatchedRecHitXY_DP1", "Number of Matched RecHits Disk +1;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_DP2] = dbe_->book2D("NMatchedRecHitXY_DP2", "Number of Matched RecHits Disk +2;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
    h_[HName::NMatchedRecHitXY_DP3] = dbe_->book2D("NMatchedRecHitXY_DP3", "Number of Matched RecHits Disk +3;X;Y", nBin, xmin, xmax, nBin, ymin, ymax);
  }

  // Residuals and pulls
  h_[HName::Res_WM2] = dbe_->book1D("Res_WM2", "Residuals for Wheel -2;Residual [cm]", 100, -8, 8);
  h_[HName::Res_WM1] = dbe_->book1D("Res_WM1", "Residuals for Wheel -1;Residual [cm]", 100, -8, 8);
  h_[HName::Res_W00] = dbe_->book1D("Res_W00", "Residuals for Wheel  0;Residual [cm]", 100, -8, 8);
  h_[HName::Res_WP1] = dbe_->book1D("Res_WP1", "Residuals for Wheel +1;Residual [cm]", 100, -8, 8);
  h_[HName::Res_WP2] = dbe_->book1D("Res_WP2", "Residuals for Wheel +2;Residual [cm]", 100, -8, 8);

  h_[HName::Res_DM3] = dbe_->book1D("Res_DM3", "Residuals for Disk -3;Residual [cm]", 100, -8, 8);
  h_[HName::Res_DM2] = dbe_->book1D("Res_DM2", "Residuals for Disk -2;Residual [cm]", 100, -8, 8);
  h_[HName::Res_DM1] = dbe_->book1D("Res_DM1", "Residuals for Disk -1;Residual [cm]", 100, -8, 8);
  h_[HName::Res_DP1] = dbe_->book1D("Res_DP1", "Residuals for Disk +1;Residual [cm]", 100, -8, 8);
  h_[HName::Res_DP2] = dbe_->book1D("Res_DP2", "Residuals for Disk +2;Residual [cm]", 100, -8, 8);
  h_[HName::Res_DP3] = dbe_->book1D("Res_DP3", "Residuals for Disk +3;Residual [cm]", 100, -8, 8);

  h_[HName::Pull_WM2] = dbe_->book1D("Pull_WM2", "Pull for Wheel -2;Pull", 100, -5, 5);
  h_[HName::Pull_WM1] = dbe_->book1D("Pull_WM1", "Pull for Wheel -1;Pull", 100, -5, 5);
  h_[HName::Pull_W00] = dbe_->book1D("Pull_W00", "Pull for Wheel  0;Pull", 100, -5, 5);
  h_[HName::Pull_WP1] = dbe_->book1D("Pull_WP1", "Pull for Wheel +1;Pull", 100, -5, 5);
  h_[HName::Pull_WP2] = dbe_->book1D("Pull_WP2", "Pull for Wheel +2;Pull", 100, -5, 5);

  h_[HName::Pull_DM3] = dbe_->book1D("Pull_DM3", "Pull for Disk -3;Pull", 100, -5, 5);
  h_[HName::Pull_DM2] = dbe_->book1D("Pull_DM2", "Pull for Disk -2;Pull", 100, -5, 5);
  h_[HName::Pull_DM1] = dbe_->book1D("Pull_DM1", "Pull for Disk -1;Pull", 100, -5, 5);
  h_[HName::Pull_DP1] = dbe_->book1D("Pull_DP1", "Pull for Disk +1;Pull", 100, -5, 5);
  h_[HName::Pull_DP2] = dbe_->book1D("Pull_DP2", "Pull for Disk +2;Pull", 100, -5, 5);
  h_[HName::Pull_DP3] = dbe_->book1D("Pull_DP3", "Pull for Disk +3;Pull", 100, -5, 5);
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
    const int station = abs(roll->id().station());
    //const int layer = roll->id().layer();
    //const int subSector = roll->id().subsector();

    if ( region == 0 ) h_[HName::NRefHit_Wheel]->Fill(ring);
    else h_[HName::NRefHit_Disk]->Fill(region*station);

    const GlobalPoint pos = roll->toGlobal(refHitIter->localPosition());
    h_[HName::RefHitEta]->Fill(pos.eta());

    if ( isStandAloneMode_ )
    {
      h_[HName::NRefHitRZ]->Fill(pos.z(), pos.perp());
      if ( region == 0 )
      {
        h_[HName::NRefHitXY_W00+ring]->Fill(pos.x(), pos.y());
      }
      else if ( region == -1 and station < 4 )
      {
        h_[HName::NRefHitXY_DM1-(station-1)]->Fill(pos.x(), pos.y());
      }
      else if ( region == 1 and station < 4 )
      {
        h_[HName::NRefHitXY_DP1+(station-1)]->Fill(pos.x(), pos.y());
      }
    }
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
    const int station = abs(roll->id().station());
    //const int layer = roll->id().layer();
    //const int subSector = roll->id().subsector();

    h_[HName::ClusterSize]->Fill(recHitIter->clusterSize());

    if ( region == 0 ) h_[HName::NRecHit_Wheel]->Fill(ring);
    else h_[HName::NRecHit_Disk]->Fill(region*station);

    const GlobalPoint pos = roll->toGlobal(recHitIter->localPosition());
    h_[HName::RecHitEta]->Fill(pos.eta());
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
    const int station = abs(roll->id().station());
    //const int layer = roll->id().layer();
    //const int subsector = roll->id().subsector();

    const double refX = refHitIter->localPosition().x();
    const double recX = recHitIter->localPosition().x();
    const double errX = recHitIter->localPositionError().xx();
    const double dX = recX - refX;
    const double pull = errX == 0 ? -999 : dX/errX;

    h_[HName::Res]->Fill(dX);
    h_[HName::Pull]->Fill(pull);

    const GlobalPoint refPos = roll->toGlobal(refHitIter->localPosition());
    const GlobalPoint recPos = roll->toGlobal(recHitIter->localPosition());

    if ( isStandAloneMode_ )
    {
      h_[HName::NMatchedRefHitRZ]->Fill(refPos.z(), refPos.perp());
      h_[HName::NMatchedRecHitRZ]->Fill(recPos.z(), recPos.perp());

      if ( region == 0 )
      {
        h_[HName::NMatchedRefHitXY_W00+ring]->Fill(refPos.x(), refPos.y());
        h_[HName::NMatchedRecHitXY_W00+ring]->Fill(recPos.x(), recPos.y());
      }
      else if ( region == -1 and station < 4 )
      {
        h_[HName::NMatchedRefHitXY_DM1-(station-1)]->Fill(refPos.x(), refPos.y());
        h_[HName::NMatchedRecHitXY_DM1-(station-1)]->Fill(recPos.x(), recPos.y());
      }
      else if ( region == 1 and station < 4 )
      {
        h_[HName::NMatchedRefHitXY_DP1+(station-1)]->Fill(refPos.x(), refPos.y());
        h_[HName::NMatchedRecHitXY_DP1+(station-1)]->Fill(recPos.x(), recPos.y());
      }
    }

    if ( region == 0 )
    {
      h_[HName::NMatchedRecHit_Wheel]->Fill(ring);
      h_[HName::Res_W00+ring]->Fill(dX);
      h_[HName::Pull_W00+ring]->Fill(pull);
    }
    else if ( region == -1 and station < 4 )
    {
      h_[HName::NMatchedRecHit_Disk]->Fill(region*station);
      h_[HName::Res_DM1-(station-1)]->Fill(dX);
      h_[HName::Pull_DM1-(station-1)]->Fill(pull);
    }
    else if ( region == 1 and station < 4 )
    {
      h_[HName::NMatchedRecHit_Disk]->Fill(region*station);
      h_[HName::Res_DP1+(station-1)]->Fill(dX);
      h_[HName::Pull_DP1+(station-1)]->Fill(pull);
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
    const int station = abs(roll->id().station());
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
      if ( region == 0 ) h_[HName::NLostHit_Wheel]->Fill(ring);
      else h_[HName::NLostHit_Disk]->Fill(region*station);
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
    const int station = abs(roll->id().station());
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
      if ( region == 0 ) h_[HName::NNoisyHit_Wheel]->Fill(ring);
      else h_[HName::NNoisyHit_Disk]->Fill(region*station);

      const GlobalPoint pos = roll->toGlobal(recHitIter->localPosition());
      h_[HName::NoisyHitEta]->Fill(pos.eta());
    }
  }
}

