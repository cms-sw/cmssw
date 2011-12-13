#include "Validation/RPCRecHits/interface/RPCRecHitValid.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

using namespace std;

typedef MonitorElement* MEP;

RPCRecHitValid::RPCRecHitValid(const edm::ParameterSet& pset)
{
  rootFileName_ = pset.getUntrackedParameter<string>("rootFileName", "");
  simHitLabel_ = pset.getParameter<edm::InputTag>("simHit");
  recHitLabel_ = pset.getParameter<edm::InputTag>("recHit");
  simTrackLabel_ = pset.getParameter<edm::InputTag>("simTrack");

  isStandAloneMode_ = pset.getUntrackedParameter<bool>("standAloneMode", false);

  dbe_ = edm::Service<DQMStore>().operator->();
  if ( !dbe_ )
  {
    edm::LogError("RPCRecHitValid") << "No DQMStore instance\n";
    return;
  }

  // Book MonitorElements
  const std::string subDir = pset.getParameter<std::string>("subDir");
  h_.bookHistograms(dbe_, subDir);

  // SimHit plots, not compatible to RPCPoint-RPCRecHit comparison
  dbe_->setCurrentFolder(subDir+"/HitProperty");
  h_simHitPType = dbe_->book1D("SimHitPType", "SimHit particle type", 11, 0, 11);
  if ( TH1F* h = h_simHitPType->getTH1F() )
  {
    h->GetXaxis()->SetBinLabel(1 , "#mu^{-}");
    h->GetXaxis()->SetBinLabel(2 , "#mu^{+}");
    h->GetXaxis()->SetBinLabel(3 , "e^{-}"  );
    h->GetXaxis()->SetBinLabel(4 , "e^{+}"  );
    h->GetXaxis()->SetBinLabel(5 , "#pi^{+}");
    h->GetXaxis()->SetBinLabel(6 , "#pi^{-}");
    h->GetXaxis()->SetBinLabel(7 , "K^{+}"  );
    h->GetXaxis()->SetBinLabel(8 , "K^{-}"  );
    h->GetXaxis()->SetBinLabel(9 , "p^{+}"  );
    h->GetXaxis()->SetBinLabel(10, "p^{-}"  );
    h->GetXaxis()->SetBinLabel(11, "Other"  );
  }

  dbe_->setCurrentFolder(subDir+"/Track");
  
  h_nRPCHitPerSimMuon        = dbe_->book1D("NRPCHitPerSimMuon"       , "Number of RPC SimHit per SimMuon", 11, -0.5, 10.5);
  h_nRPCHitPerSimMuonBarrel  = dbe_->book1D("NRPCHitPerSimMuonBarrel" , "Number of RPC SimHit per SimMuon", 11, -0.5, 10.5);
  h_nRPCHitPerSimMuonOverlap = dbe_->book1D("NRPCHitPerSimMuonOverlap", "Number of RPC SimHit per SimMuon", 11, -0.5, 10.5);
  h_nRPCHitPerSimMuonEndcap  = dbe_->book1D("NRPCHitPerSimMuonEndcap" , "Number of RPC SimHit per SimMuon", 11, -0.5, 10.5);

  float ptBins[] = {0, 1, 2, 5, 10, 20, 30, 50, 100, 200, 300, 500};
  const int nPtBins = sizeof(ptBins)/sizeof(float)-1;
  h_simMuonBarrel_pt   = dbe_->book1D("SimMuonBarrel_pt"  , "SimMuon RPCHit in Barrel  p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_simMuonOverlap_pt  = dbe_->book1D("SimMuonOverlap_pt" , "SimMuon RPCHit in Overlap p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_simMuonEndcap_pt   = dbe_->book1D("SimMuonEndcap_pt"  , "SimMuon RPCHit in Endcap  p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_simMuonNoRPC_pt  = dbe_->book1D("SimMuonNoRPC_pt" , "SimMuon without RPCHit p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_simMuonBarrel_eta  = dbe_->book1D("SimMuonBarrel_eta" , "SimMuon RPCHit in Barrel  #eta;#eta", 50, -2.5, 2.5);
  h_simMuonOverlap_eta = dbe_->book1D("SimMuonOverlap_eta", "SimMuon RPCHit in Overlap #eta;#eta", 50, -2.5, 2.5);
  h_simMuonEndcap_eta  = dbe_->book1D("SimMuonEndcap_eta" , "SimMuon RPCHit in Endcap  #eta;#eta", 50, -2.5, 2.5);
  h_simMuonNoRPC_eta = dbe_->book1D("SimMuonNoRPC_eta", "SimMuon without RPCHit #eta;#eta", 50, -2.5, 2.5);

  dbe_->setCurrentFolder(subDir+"/Occupancy");
  
  h_refBkgBarrelOccupancy_wheel   = dbe_->book1D("BkgBarrelOccupancy_wheel"  , "Bkg occupancy", 5, -2.5, 2.5);
  h_refBkgEndcapOccupancy_disk    = dbe_->book1D("BkgEndcapOccupancy_disk"   , "Bkg occupancy", 7, -3.5, 3.5);
  h_refBkgBarrelOccupancy_station = dbe_->book1D("BkgBarrelOccupancy_station", "Bkg occupancy", 4,  0.5, 4.5);
  h_refPunchBarrelOccupancy_wheel   = dbe_->book1D("RefPunchBarrelOccupancy_wheel"  , "RefPunchthrough occupancy", 5, -2.5, 2.5);
  h_refPunchEndcapOccupancy_disk    = dbe_->book1D("RefPunchEndcapOccupancy_disk"   , "RefPunchthrough occupancy", 7, -3.5, 3.5);
  h_refPunchBarrelOccupancy_station = dbe_->book1D("RefPunchBarrelOccupancy_station", "RefPunchthrough occupancy", 4,  0.5, 4.5);
  h_recPunchBarrelOccupancy_wheel   = dbe_->book1D("RecPunchBarrelOccupancy_wheel"  , "Punchthrough recHit occupancy", 5, -2.5, 2.5);
  h_recPunchEndcapOccupancy_disk    = dbe_->book1D("RecPunchEndcapOccupancy_disk"   , "Punchthrough recHit occupancy", 7, -3.5, 3.5);
  h_recPunchBarrelOccupancy_station = dbe_->book1D("RecPunchBarrelOccupancy_station", "Punchthrough recHit occupancy", 4,  0.5, 4.5);
  h_noiseBarrelOccupancy_wheel   = dbe_->book1D("NoiseBarrelOccupancy_wheel"  , "Noise recHit occupancy", 5, -2.5, 2.5);
  h_noiseEndcapOccupancy_disk    = dbe_->book1D("NoiseEndcapOccupancy_disk"   , "Noise recHit occupancy", 7, -3.5, 3.5);
  h_noiseBarrelOccupancy_station = dbe_->book1D("NoiseBarrelOccupancy_station", "Noise recHit occupancy", 4,  0.5, 4.5);

  h_refBkgBarrelOccupancy_wheel_station = dbe_->book2D("BkgBarrelOccupancy_wheel_station", "Bkg occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  h_refBkgEndcapOccupancy_disk_ring     = dbe_->book2D("BkgEndcapOccupancy_disk_ring"    , "Bkg occupancy", 7, -3.5, 3.5, 4, 0.5, 4.5);
  h_refPunchBarrelOccupancy_wheel_station = dbe_->book2D("RefPunchBarrelOccupancy_wheel_station", "RefPunchthrough occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  h_refPunchEndcapOccupancy_disk_ring     = dbe_->book2D("RefPunchEndcapOccupancy_disk_ring"    , "RefPunchthrough occupancy", 7, -3.5, 3.5, 4, 0.5, 4.5);
  h_recPunchBarrelOccupancy_wheel_station = dbe_->book2D("RecPunchBarrelOccupancy_wheel_station", "Punchthrough recHit occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  h_recPunchEndcapOccupancy_disk_ring     = dbe_->book2D("RecPunchEndcapOccupancy_disk_ring"    , "Punchthrough recHit occupancy", 7, -3.5, 3.5, 4, 0.5, 4.5);
  h_noiseBarrelOccupancy_wheel_station = dbe_->book2D("NoiseBarrelOccupancy_wheel_station", "Noise recHit occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  h_noiseEndcapOccupancy_disk_ring     = dbe_->book2D("NoiseEndcapOccupancy_disk_ring"    , "Noise recHit occupancy", 7, -3.5, 3.5, 4, 0.5, 4.5);

  h_refBkgBarrelOccupancy_wheel_station->getTH2F()->SetOption("COLZ");
  h_refBkgEndcapOccupancy_disk_ring    ->getTH2F()->SetOption("COLZ");
  h_refPunchBarrelOccupancy_wheel_station->getTH2F()->SetOption("COLZ");
  h_refPunchEndcapOccupancy_disk_ring    ->getTH2F()->SetOption("COLZ");
  h_recPunchBarrelOccupancy_wheel_station->getTH2F()->SetOption("COLZ");
  h_recPunchEndcapOccupancy_disk_ring    ->getTH2F()->SetOption("COLZ");
  h_noiseBarrelOccupancy_wheel_station->getTH2F()->SetOption("COLZ");
  h_noiseEndcapOccupancy_disk_ring    ->getTH2F()->SetOption("COLZ");

  for ( int i=1; i<=5; ++i )
  {
    TString binLabel = Form("Wheel %d", i-3);
    h_refBkgBarrelOccupancy_wheel->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refBkgBarrelOccupancy_wheel_station->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refPunchBarrelOccupancy_wheel->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refPunchBarrelOccupancy_wheel_station->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchBarrelOccupancy_wheel->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchBarrelOccupancy_wheel_station->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_noiseBarrelOccupancy_wheel->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_noiseBarrelOccupancy_wheel_station->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=7; ++i )
  {
    TString binLabel = Form("Disk %d", i-4);
    h_refBkgEndcapOccupancy_disk  ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refBkgEndcapOccupancy_disk_ring  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refPunchEndcapOccupancy_disk  ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refPunchEndcapOccupancy_disk_ring  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchEndcapOccupancy_disk  ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchEndcapOccupancy_disk_ring  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_noiseEndcapOccupancy_disk  ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_noiseEndcapOccupancy_disk_ring  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=4; ++i )
  {
    TString binLabel = Form("Station %d", i);
    h_refBkgBarrelOccupancy_station  ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refBkgBarrelOccupancy_wheel_station  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    h_refPunchBarrelOccupancy_station  ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refPunchBarrelOccupancy_wheel_station  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    h_recPunchBarrelOccupancy_station  ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchBarrelOccupancy_wheel_station  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    h_noiseBarrelOccupancy_station  ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_noiseBarrelOccupancy_wheel_station  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=4; ++i )
  {
    TString binLabel = Form("Ring %d", i);
    h_refBkgEndcapOccupancy_disk_ring  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    h_refPunchEndcapOccupancy_disk_ring  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    h_recPunchEndcapOccupancy_disk_ring  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    h_noiseEndcapOccupancy_disk_ring  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
  }
}

RPCRecHitValid::~RPCRecHitValid()
{
  if ( dbe_ )
  {
    if ( !rootFileName_.empty() ) dbe_->save(rootFileName_);
  }
}

void RPCRecHitValid::beginJob()
{
}

void RPCRecHitValid::endJob()
{
}

void RPCRecHitValid::analyze(const edm::Event& event, const edm::EventSetup& eventSetup)
{
  if ( !dbe_ )
  {
    edm::LogError("RPCRecHitValid") << "No DQMStore instance\n";
    return;
  }

  // Get the RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);

  // Retrieve SimHits from the event
  edm::Handle<edm::PSimHitContainer> simHitHandle;
  if ( !event.getByLabel(simHitLabel_, simHitHandle) )
  {
    edm::LogInfo("RPCRecHitValid") << "Cannot find simHit collection\n";
    return;
  }

  // Retrieve RecHits from the event
  edm::Handle<RPCRecHitCollection> recHitHandle;
  if ( !event.getByLabel(recHitLabel_, recHitHandle) )
  {
    edm::LogInfo("RPCRecHitValid") << "Cannot find recHit collection\n";
    return;
  }

  // Get SimTracks
  edm::Handle<edm::View<TrackingParticle> > simTrackHandle;
  if ( !event.getByLabel(simTrackLabel_, simTrackHandle) )
  {
    edm::LogInfo("RPCRecHitValid") << "Cannot find simTrack collection\n";
    return;
  }

  typedef edm::PSimHitContainer::const_iterator SimHitIter;
  typedef RPCRecHitCollection::const_iterator RecHitIter;

  for ( edm::View<TrackingParticle>::const_iterator simTrack = simTrackHandle->begin();
        simTrack != simTrackHandle->end(); ++simTrack )
  {
    if ( abs(simTrack->pdgId()) != 13 ) continue;
    int nRPCBarrelHit = 0;
    int nRPCEndcapHit = 0;

    for ( SimHitIter simHit = simTrack->pSimHit_begin();
          simHit != simTrack->pSimHit_end(); ++simHit )
    {
      const DetId detId(simHit->detUnitId());
      if ( detId.det() != DetId::Muon or detId.subdetId() != MuonSubdetId::RPC ) continue;
      const RPCDetId rpcDetId = static_cast<const RPCDetId>(simHit->detUnitId());
      const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(rpcDetId()));
      if ( !roll ) continue;

      const int region = roll->id().region();
      //const int ring = roll->id().ring();
      //const int sector = roll->id().sector();
      //const int station = roll->id().station();
      //const int layer = roll->id().layer();
      //const int subSector = roll->id().subsector();

      if ( region == 0 ) ++nRPCBarrelHit;
      else ++nRPCEndcapHit;
    }

    int nRPCHit = nRPCBarrelHit+nRPCEndcapHit;
    h_nRPCHitPerSimMuon->Fill(nRPCHit);
    if ( nRPCBarrelHit and nRPCEndcapHit )
    {
      h_nRPCHitPerSimMuonOverlap->Fill(nRPCHit);
      h_simMuonOverlap_pt->Fill(simTrack->pt());
      h_simMuonOverlap_eta->Fill(simTrack->eta());
    }
    else if ( nRPCBarrelHit )
    {
      h_nRPCHitPerSimMuonBarrel->Fill(nRPCHit);
      h_simMuonBarrel_pt->Fill(simTrack->pt());
      h_simMuonBarrel_eta->Fill(simTrack->eta());
    }
    else if ( nRPCEndcapHit )
    {
      h_nRPCHitPerSimMuonEndcap->Fill(nRPCHit);
      h_simMuonEndcap_pt->Fill(simTrack->pt());
      h_simMuonEndcap_eta->Fill(simTrack->eta());
    }
    else
    {
      h_simMuonNoRPC_pt->Fill(simTrack->pt());
      h_simMuonNoRPC_eta->Fill(simTrack->eta());
    }
  }

  // Loop over simHits, fill histograms which does not need associations
  int nRefHitBarrel = 0, nRefHitEndcap = 0;
  for ( SimHitIter simHitIter = simHitHandle->begin();
        simHitIter != simHitHandle->end(); ++simHitIter )
  {
    const RPCDetId detId = static_cast<const RPCDetId>(simHitIter->detUnitId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId()));
    if ( !roll ) continue;

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subSector = roll->id().subsector();

    const int simHitPType = simHitIter->particleType();
    switch ( simHitPType )
    {
      case  13:
        h_simHitPType->Fill(0);
        break;
      case -13:
        h_simHitPType->Fill(1);
        break;
      case  11:
        h_simHitPType->Fill(2);
        break;
      case -11:
        h_simHitPType->Fill(3);
        break;
      case  211:
        h_simHitPType->Fill(4);
        break;
      case -211:
        h_simHitPType->Fill(5);
        break;
      case  321:
        h_simHitPType->Fill(6);
        break;
      case -321:
        h_simHitPType->Fill(7);
        break;
      case  2212:
        h_simHitPType->Fill(8);
        break;
      case -2212:
        h_simHitPType->Fill(9);
        break;
      default:
        h_simHitPType->Fill(10);
        break;
    }

    const int absSimHitPType = abs(simHitPType);
    if ( absSimHitPType == 13 )
    {
      if ( region == 0 ) 
      {
        ++nRefHitBarrel;
        h_.refHitBarrelOccupancy_wheel->Fill(ring);
        h_.refHitBarrelOccupancy_station->Fill(station);
        h_.refHitBarrelOccupancy_wheel_station->Fill(ring, station);
      }
      else 
      {
        ++nRefHitEndcap;
        h_.refHitEndcapOccupancy_disk->Fill(region*station);
        h_.refHitEndcapOccupancy_disk_ring->Fill(region*station, ring);
      }
    }
    else if ( absSimHitPType == 211 or absSimHitPType == 321 or absSimHitPType == 2212 )
    {
      if ( region == 0 )
      {
        h_refPunchBarrelOccupancy_wheel->Fill(ring);
        h_refPunchBarrelOccupancy_station->Fill(station);
        h_refPunchBarrelOccupancy_wheel_station->Fill(ring, station);
      }
      else
      {
        h_refPunchEndcapOccupancy_disk->Fill(region*station);
        h_refPunchEndcapOccupancy_disk_ring->Fill(region*station, ring);
      }
    }
    else
    {
      if ( region == 0 )
      {
        h_refBkgBarrelOccupancy_wheel->Fill(ring);
        h_refBkgBarrelOccupancy_station->Fill(station);
        h_refBkgBarrelOccupancy_wheel_station->Fill(ring, station);
      }
      else
      {
        h_refBkgEndcapOccupancy_disk->Fill(region*station);
        h_refBkgEndcapOccupancy_disk_ring->Fill(region*station, ring);
      }
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
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subSector = roll->id().subsector();

    h_.clusterSize->Fill(recHitIter->clusterSize());

    if ( region == 0 ) 
    {
      ++nRecHitBarrel;
      sumClusterSizeBarrel += recHitIter->clusterSize();
      h_.clusterSizeBarrel->Fill(recHitIter->clusterSize());
      h_.recHitBarrelOccupancy_wheel->Fill(ring);
      h_.recHitBarrelOccupancy_station->Fill(station);
      h_.recHitBarrelOccupancy_wheel_station->Fill(ring, station);
    }
    else
    {
      ++nRecHitEndcap;
      sumClusterSizeEndcap += recHitIter->clusterSize();
      h_.clusterSizeEndcap->Fill(recHitIter->clusterSize());
      h_.recHitEndcapOccupancy_disk->Fill(region*station);
      h_.recHitEndcapOccupancy_disk_ring->Fill(region*station, ring);
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

  // Start matching SimHits to RecHits
  typedef std::map<SimHitIter, RecHitIter> SimToRecHitMap;
  SimToRecHitMap simToRecHitMap;

  for ( SimHitIter simHitIter = simHitHandle->begin();
        simHitIter != simHitHandle->end(); ++simHitIter )
  {
    if ( abs(simHitIter->particleType()) != 13 ) continue;

    const RPCDetId simDetId = static_cast<const RPCDetId>(simHitIter->detUnitId());
    const RPCRoll* simRoll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(simDetId));
    if ( !simRoll ) continue;

    const double simX = simHitIter->localPosition().x();

    for ( RecHitIter recHitIter = recHitHandle->begin();
          recHitIter != recHitHandle->end(); ++recHitIter )
    {
      const RPCDetId recDetId = static_cast<const RPCDetId>(recHitIter->rpcId());
      const RPCRoll* recRoll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(recDetId));
      if ( !recRoll ) continue;

      if ( simDetId != recDetId ) continue;

      const double recX = recHitIter->localPosition().x();
      const double newDx = fabs(recX - simX);

      // Associate SimHit to RecHit
      SimToRecHitMap::const_iterator prevSimToReco = simToRecHitMap.find(simHitIter);
      if ( prevSimToReco == simToRecHitMap.end() )
      {
        simToRecHitMap.insert(std::make_pair(simHitIter, recHitIter));
      }
      else
      {
        const double oldDx = fabs(prevSimToReco->second->localPosition().x() - simX);

        if ( newDx < oldDx )
        {
          simToRecHitMap[simHitIter] = recHitIter;
        }
      }
    }
  }

  // Now we have simHit-recHit mapping
  // So we can fill up relavant histograms
  int nMatchHitBarrel = 0, nMatchHitEndcap = 0;
  for ( SimToRecHitMap::const_iterator match = simToRecHitMap.begin();
        match != simToRecHitMap.end(); ++match )
  {
    SimHitIter simHitIter = match->first;
    RecHitIter recHitIter = match->second;

    const RPCDetId detId = static_cast<const RPCDetId>(simHitIter->detUnitId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId));

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subsector = roll->id().subsector();

    const double simX = simHitIter->localPosition().x();
    const double recX = recHitIter->localPosition().x();
    const double errX = sqrt(recHitIter->localPositionError().xx());
    const double dX = recX - simX;
    const double pull = errX == 0 ? -999 : dX/errX;
  
    //const GlobalPoint simPos = roll->toGlobal(simHitIter->localPosition());
    //const GlobalPoint recPos = roll->toGlobal(recHitIter->localPosition());

    if ( region == 0 )
    {
      ++nMatchHitBarrel;
      h_.resBarrel->Fill(dX);
      h_.pullBarrel->Fill(pull);
      h_.matchBarrelOccupancy_wheel->Fill(ring);
      h_.matchBarrelOccupancy_station->Fill(station);
      h_.matchBarrelOccupancy_wheel_station->Fill(ring, station);

      h_.res_wheel_res->Fill(ring, dX);
      h_.res_station_res->Fill(station, dX);
      h_.pull_wheel_pull->Fill(ring, pull);
      h_.pull_station_pull->Fill(station, pull);
    }
    else
    {
      ++nMatchHitEndcap;
      h_.resEndcap->Fill(dX);
      h_.pullEndcap->Fill(pull);
      h_.matchEndcapOccupancy_disk->Fill(region*station);
      h_.matchEndcapOccupancy_disk_ring->Fill(region*station, ring);

      h_.res_disk_res->Fill(region*station, dX);
      h_.res_ring_res->Fill(ring, dX);
      h_.pull_disk_pull->Fill(region*station, pull);
      h_.pull_ring_pull->Fill(ring, pull);
    }

  }
  h_.nMatchHitBarrel->Fill(nMatchHitBarrel);
  h_.nMatchHitEndcap->Fill(nMatchHitEndcap);
/*
  // Find Lost hits
  for ( SimHitIter simHitIter = simHitHandle->begin();
        simHitIter != simHitHandle->end(); ++simHitIter )
  {
    const RPCDetId detId = static_cast<const RPCDetId>(simHitIter->detUnitId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId));

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subsector = roll->id().subsector();

    bool matched = false;
    for ( SimToRecHitMap::const_iterator match = simToRecHitMap.begin();
          match != simToRecHitMap.end(); ++match )
    {
      if ( simHitIter == match->first )
      {
        matched = true;
        break;
      }
    }

    if ( !matched )
    {
      if ( region == 0 )
      {
        h_.nUmatchBarrelOccupancy_wheel->Fill(ring);
        h_.nUmatchBarrelOccupancy_wheelvsR->Fill(ring, station);
      }
      else
      {
        h_.nUmatchBarrelOccupancy_disk->Fill(region*station);
        h_.nUmatchBarrelOccupancy_diskvsR->Fill(region*station, ring);
      }
    }
  }
*/

  // Find Non-muon hits
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
    for ( SimToRecHitMap::const_iterator match = simToRecHitMap.begin();
          match != simToRecHitMap.end(); ++match )
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
        h_.umBarrelOccupancy_wheel->Fill(ring);
        h_.umBarrelOccupancy_station->Fill(station);
        h_.umBarrelOccupancy_wheel_station->Fill(ring, station);
      }
      else
      {
        h_.umEndcapOccupancy_disk->Fill(region*station);
        h_.umEndcapOccupancy_disk_ring->Fill(region*station, ring);
      }

//      const GlobalPoint pos = roll->toGlobal(recHitIter->localPosition());
//      h_[HName::NoisyHitEta]->Fill(pos.eta());

      int nPunchMatched = 0;
      int nNonMuMatched = 0;
      // Check if this recHit came from non-muon simHit
      for ( SimHitIter simHitIter = simHitHandle->begin();
            simHitIter != simHitHandle->end(); ++simHitIter )
      {
        const int absSimHitPType = abs(simHitIter->particleType());
        if ( absSimHitPType == 13 ) continue;

        const RPCDetId simDetId = static_cast<const RPCDetId>(simHitIter->detUnitId());
        if ( simDetId == detId )
        {
          if ( absSimHitPType == 211 or absSimHitPType == 321 or absSimHitPType == 2212 )
          {
            ++nPunchMatched;
          }
          ++nNonMuMatched;
        }
      }

      if ( nPunchMatched > 0 )
      {
        if ( region == 0 )
        {
          h_recPunchBarrelOccupancy_wheel->Fill(ring);
          h_recPunchBarrelOccupancy_station->Fill(station);
          h_recPunchBarrelOccupancy_wheel_station->Fill(ring, station);
        }
        else
        {
          h_recPunchEndcapOccupancy_disk->Fill(region*station);
          h_recPunchEndcapOccupancy_disk_ring->Fill(region*station, ring);
        }
      }
      else if ( nNonMuMatched == 0 ) // No matches found
      {
        if ( region == 0 )
        {
          h_noiseBarrelOccupancy_wheel->Fill(ring);
          h_noiseBarrelOccupancy_station->Fill(station);
          h_noiseBarrelOccupancy_wheel_station->Fill(ring, station);
        }
        else
        {
          h_noiseEndcapOccupancy_disk->Fill(region*station);
          h_noiseEndcapOccupancy_disk_ring->Fill(region*station, ring);
        }
      }

/*
      if ( nPunchMatched > 1 or nNonMuMatched > 1 ) 
      {
        cout << nPunchMatched << ' ' << nNonMuMatched << endl;
        cout << region << ' ' << ring << ' ' << station << endl;
        cout << "----" << endl;
      }
*/
    }
  }
}

DEFINE_FWK_MODULE(RPCRecHitValid);

