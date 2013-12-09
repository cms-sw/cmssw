#include "Validation/RPCRecHits/interface/RPCRecHitValid.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"

#include <algorithm>

using namespace std;

typedef MonitorElement* MEP;

RPCRecHitValid::RPCRecHitValid(const edm::ParameterSet& pset)
{
  simHitLabel_ = pset.getParameter<edm::InputTag>("simHit");
  recHitLabel_ = pset.getParameter<edm::InputTag>("recHit");
  simParticleLabel_ = pset.getParameter<edm::InputTag>("simTrack");
  simHitAssocLabel_ = pset.getParameter<edm::InputTag>("simHitAssoc");
  muonLabel_ = pset.getParameter<edm::InputTag>("muon");
  dbe_ = edm::Service<DQMStore>().operator->();
  if ( !dbe_ )
  {
    edm::LogError("RPCRecHitValid") << "No DQMStore instance\n";
    return;
  }

  // Book MonitorElements
  subDir_ = pset.getParameter<std::string>("subDir");
  h_.bookHistograms(dbe_, subDir_);

  // SimHit plots, not compatible to RPCPoint-RPCRecHit comparison
  dbe_->setCurrentFolder(subDir_+"/HitProperty");
  h_simParticleType = dbe_->book1D("SimHitPType", "SimHit particle type", 11, 0, 11);
  h_simParticleType->getTH1()->SetMinimum(0);
  if ( TH1* h = h_simParticleType->getTH1() )
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

  dbe_->setCurrentFolder(subDir_+"/Track");

  h_nRPCHitPerSimMuon        = dbe_->book1D("NRPCHitPerSimMuon"       , "Number of RPC SimHit per SimMuon", 11, -0.5, 10.5);
  h_nRPCHitPerSimMuonBarrel  = dbe_->book1D("NRPCHitPerSimMuonBarrel" , "Number of RPC SimHit per SimMuon", 11, -0.5, 10.5);
  h_nRPCHitPerSimMuonOverlap = dbe_->book1D("NRPCHitPerSimMuonOverlap", "Number of RPC SimHit per SimMuon", 11, -0.5, 10.5);
  h_nRPCHitPerSimMuonEndcap  = dbe_->book1D("NRPCHitPerSimMuonEndcap" , "Number of RPC SimHit per SimMuon", 11, -0.5, 10.5);

  h_nRPCHitPerRecoMuon        = dbe_->book1D("NRPCHitPerRecoMuon"       , "Number of RPC RecHit per RecoMuon", 11, -0.5, 10.5);
  h_nRPCHitPerRecoMuonBarrel  = dbe_->book1D("NRPCHitPerRecoMuonBarrel" , "Number of RPC RecHit per RecoMuon", 11, -0.5, 10.5);
  h_nRPCHitPerRecoMuonOverlap = dbe_->book1D("NRPCHitPerRecoMuonOverlap", "Number of RPC RecHit per RecoMuon", 11, -0.5, 10.5);
  h_nRPCHitPerRecoMuonEndcap  = dbe_->book1D("NRPCHitPerRecoMuonEndcap" , "Number of RPC RecHit per RecoMuon", 11, -0.5, 10.5);

  h_nRPCHitPerSimMuon        ->getTH1()->SetMinimum(0);
  h_nRPCHitPerSimMuonBarrel  ->getTH1()->SetMinimum(0);
  h_nRPCHitPerSimMuonOverlap ->getTH1()->SetMinimum(0);
  h_nRPCHitPerSimMuonEndcap  ->getTH1()->SetMinimum(0);

  h_nRPCHitPerRecoMuon       ->getTH1()->SetMinimum(0);
  h_nRPCHitPerRecoMuonBarrel ->getTH1()->SetMinimum(0);
  h_nRPCHitPerRecoMuonOverlap->getTH1()->SetMinimum(0);
  h_nRPCHitPerRecoMuonEndcap ->getTH1()->SetMinimum(0);

  float ptBins[] = {0, 1, 2, 5, 10, 20, 30, 50, 100, 200, 300, 500};
  const int nPtBins = sizeof(ptBins)/sizeof(float)-1;
  h_simMuonBarrel_pt   = dbe_->book1D("SimMuonBarrel_pt"  , "SimMuon RPCHit in Barrel  p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_simMuonOverlap_pt  = dbe_->book1D("SimMuonOverlap_pt" , "SimMuon RPCHit in Overlap p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_simMuonEndcap_pt   = dbe_->book1D("SimMuonEndcap_pt"  , "SimMuon RPCHit in Endcap  p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_simMuonNoRPC_pt    = dbe_->book1D("SimMuonNoRPC_pt" , "SimMuon without RPCHit p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_simMuonBarrel_eta  = dbe_->book1D("SimMuonBarrel_eta" , "SimMuon RPCHit in Barrel  #eta;#eta", 50, -2.5, 2.5);
  h_simMuonOverlap_eta = dbe_->book1D("SimMuonOverlap_eta", "SimMuon RPCHit in Overlap #eta;#eta", 50, -2.5, 2.5);
  h_simMuonEndcap_eta  = dbe_->book1D("SimMuonEndcap_eta" , "SimMuon RPCHit in Endcap  #eta;#eta", 50, -2.5, 2.5);
  h_simMuonNoRPC_eta   = dbe_->book1D("SimMuonNoRPC_eta", "SimMuon without RPCHit #eta;#eta", 50, -2.5, 2.5);
  h_simMuonBarrel_phi  = dbe_->book1D("SimMuonBarrel_phi" , "SimMuon RPCHit in Barrel  #phi;#phi", 36, -TMath::Pi(), TMath::Pi());
  h_simMuonOverlap_phi = dbe_->book1D("SimMuonOverlap_phi", "SimMuon RPCHit in Overlap #phi;#phi", 36, -TMath::Pi(), TMath::Pi());
  h_simMuonEndcap_phi  = dbe_->book1D("SimMuonEndcap_phi" , "SimMuon RPCHit in Endcap  #phi;#phi", 36, -TMath::Pi(), TMath::Pi());
  h_simMuonNoRPC_phi   = dbe_->book1D("SimMuonNoRPC_phi", "SimMuon without RPCHit #phi;#phi", 36, -TMath::Pi(), TMath::Pi());

  h_recoMuonBarrel_pt   = dbe_->book1D("RecoMuonBarrel_pt"  , "RecoMuon RPCHit in Barrel  p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_recoMuonOverlap_pt  = dbe_->book1D("RecoMuonOverlap_pt" , "RecoMuon RPCHit in Overlap p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_recoMuonEndcap_pt   = dbe_->book1D("RecoMuonEndcap_pt"  , "RecoMuon RPCHit in Endcap  p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_recoMuonNoRPC_pt    = dbe_->book1D("RecoMuonNoRPC_pt" , "RecoMuon without RPCHit p_{T};p_{T} [GeV/c^{2}]", nPtBins, ptBins);
  h_recoMuonBarrel_eta  = dbe_->book1D("RecoMuonBarrel_eta" , "RecoMuon RPCHit in Barrel  #eta;#eta", 50, -2.5, 2.5);
  h_recoMuonOverlap_eta = dbe_->book1D("RecoMuonOverlap_eta", "RecoMuon RPCHit in Overlap #eta;#eta", 50, -2.5, 2.5);
  h_recoMuonEndcap_eta  = dbe_->book1D("RecoMuonEndcap_eta" , "RecoMuon RPCHit in Endcap  #eta;#eta", 50, -2.5, 2.5);
  h_recoMuonNoRPC_eta   = dbe_->book1D("RecoMuonNoRPC_eta", "RecoMuon without RPCHit #eta;#eta", 50, -2.5, 2.5);
  h_recoMuonBarrel_phi  = dbe_->book1D("RecoMuonBarrel_phi" , "RecoMuon RPCHit in Barrel  #phi;#phi", 36, -TMath::Pi(), TMath::Pi());
  h_recoMuonOverlap_phi = dbe_->book1D("RecoMuonOverlap_phi", "RecoMuon RPCHit in Overlap #phi;#phi", 36, -TMath::Pi(), TMath::Pi());
  h_recoMuonEndcap_phi  = dbe_->book1D("RecoMuonEndcap_phi" , "RecoMuon RPCHit in Endcap  #phi;#phi", 36, -TMath::Pi(), TMath::Pi());
  h_recoMuonNoRPC_phi   = dbe_->book1D("RecoMuonNoRPC_phi", "RecoMuon without RPCHit #phi;#phi", 36, -TMath::Pi(), TMath::Pi());

  h_simMuonBarrel_pt   ->getTH1()->SetMinimum(0);
  h_simMuonOverlap_pt  ->getTH1()->SetMinimum(0);
  h_simMuonEndcap_pt   ->getTH1()->SetMinimum(0);
  h_simMuonNoRPC_pt    ->getTH1()->SetMinimum(0);
  h_simMuonBarrel_eta  ->getTH1()->SetMinimum(0);
  h_simMuonOverlap_eta ->getTH1()->SetMinimum(0);
  h_simMuonEndcap_eta  ->getTH1()->SetMinimum(0);
  h_simMuonNoRPC_eta   ->getTH1()->SetMinimum(0);
  h_simMuonBarrel_phi  ->getTH1()->SetMinimum(0);
  h_simMuonOverlap_phi ->getTH1()->SetMinimum(0);
  h_simMuonEndcap_phi  ->getTH1()->SetMinimum(0);
  h_simMuonNoRPC_phi   ->getTH1()->SetMinimum(0);

  h_recoMuonBarrel_pt  ->getTH1()->SetMinimum(0);
  h_recoMuonOverlap_pt ->getTH1()->SetMinimum(0);
  h_recoMuonEndcap_pt  ->getTH1()->SetMinimum(0);
  h_recoMuonNoRPC_pt   ->getTH1()->SetMinimum(0);
  h_recoMuonBarrel_eta ->getTH1()->SetMinimum(0);
  h_recoMuonOverlap_eta->getTH1()->SetMinimum(0);
  h_recoMuonEndcap_eta ->getTH1()->SetMinimum(0);
  h_recoMuonNoRPC_eta  ->getTH1()->SetMinimum(0);
  h_recoMuonBarrel_phi ->getTH1()->SetMinimum(0);
  h_recoMuonOverlap_phi->getTH1()->SetMinimum(0);
  h_recoMuonEndcap_phi ->getTH1()->SetMinimum(0);
  h_recoMuonNoRPC_phi  ->getTH1()->SetMinimum(0);

  dbe_->setCurrentFolder(subDir_+"/Occupancy");

  h_eventCount = dbe_->book1D("EventCount", "Event count", 3, 1, 4);
  h_eventCount->getTH1()->SetMinimum(0);
  if ( h_eventCount )
  {
    TH1* h = h_eventCount->getTH1();
    h->GetXaxis()->SetBinLabel(1, "eventBegin");
    h->GetXaxis()->SetBinLabel(2, "eventEnd");
    h->GetXaxis()->SetBinLabel(3, "run");
  }

  h_refPunchOccupancyBarrel_wheel   = dbe_->book1D("RefPunchOccupancyBarrel_wheel"  , "RefPunchthrough occupancy", 5, -2.5, 2.5);
  h_refPunchOccupancyEndcap_disk    = dbe_->book1D("RefPunchOccupancyEndcap_disk"   , "RefPunchthrough occupancy", 9, -4.5, 4.5);
  h_refPunchOccupancyBarrel_station = dbe_->book1D("RefPunchOccupancyBarrel_station", "RefPunchthrough occupancy", 4,  0.5, 4.5);
  h_recPunchOccupancyBarrel_wheel   = dbe_->book1D("RecPunchOccupancyBarrel_wheel"  , "Punchthrough recHit occupancy", 5, -2.5, 2.5);
  h_recPunchOccupancyEndcap_disk    = dbe_->book1D("RecPunchOccupancyEndcap_disk"   , "Punchthrough recHit occupancy", 9, -4.5, 4.5);
  h_recPunchOccupancyBarrel_station = dbe_->book1D("RecPunchOccupancyBarrel_station", "Punchthrough recHit occupancy", 4,  0.5, 4.5);

  h_refPunchOccupancyBarrel_wheel   ->getTH1()->SetMinimum(0);
  h_refPunchOccupancyEndcap_disk    ->getTH1()->SetMinimum(0);
  h_refPunchOccupancyBarrel_station ->getTH1()->SetMinimum(0);
  h_recPunchOccupancyBarrel_wheel   ->getTH1()->SetMinimum(0);
  h_recPunchOccupancyEndcap_disk    ->getTH1()->SetMinimum(0);
  h_recPunchOccupancyBarrel_station ->getTH1()->SetMinimum(0);

  h_refPunchOccupancyBarrel_wheel_station = dbe_->book2D("RefPunchOccupancyBarrel_wheel_station", "RefPunchthrough occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  h_refPunchOccupancyEndcap_disk_ring     = dbe_->book2D("RefPunchOccupancyEndcap_disk_ring"    , "RefPunchthrough occupancy", 9, -4.5, 4.5, 4, 0.5, 4.5);
  h_recPunchOccupancyBarrel_wheel_station = dbe_->book2D("RecPunchOccupancyBarrel_wheel_station", "Punchthrough recHit occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  h_recPunchOccupancyEndcap_disk_ring     = dbe_->book2D("RecPunchOccupancyEndcap_disk_ring"    , "Punchthrough recHit occupancy", 9, -4.5, 4.5, 4, 0.5, 4.5);

  h_refPunchOccupancyBarrel_wheel_station->getTH2F()->SetOption("COLZ");
  h_refPunchOccupancyEndcap_disk_ring    ->getTH2F()->SetOption("COLZ");
  h_recPunchOccupancyBarrel_wheel_station->getTH2F()->SetOption("COLZ");
  h_recPunchOccupancyEndcap_disk_ring    ->getTH2F()->SetOption("COLZ");

  h_refPunchOccupancyBarrel_wheel_station->getTH2F()->SetContour(10);
  h_refPunchOccupancyEndcap_disk_ring    ->getTH2F()->SetContour(10);
  h_recPunchOccupancyBarrel_wheel_station->getTH2F()->SetContour(10);
  h_recPunchOccupancyEndcap_disk_ring    ->getTH2F()->SetContour(10);

  h_refPunchOccupancyBarrel_wheel_station->getTH2F()->SetStats(0);
  h_refPunchOccupancyEndcap_disk_ring    ->getTH2F()->SetStats(0);
  h_recPunchOccupancyBarrel_wheel_station->getTH2F()->SetStats(0);
  h_recPunchOccupancyEndcap_disk_ring    ->getTH2F()->SetStats(0);

  h_refPunchOccupancyBarrel_wheel_station->getTH2F()->SetMinimum(0);
  h_refPunchOccupancyEndcap_disk_ring    ->getTH2F()->SetMinimum(0);
  h_recPunchOccupancyBarrel_wheel_station->getTH2F()->SetMinimum(0);
  h_recPunchOccupancyEndcap_disk_ring    ->getTH2F()->SetMinimum(0);

  for ( int i=1; i<=5; ++i )
  {
    TString binLabel = Form("Wheel %d", i-3);
    h_refPunchOccupancyBarrel_wheel->getTH1()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refPunchOccupancyBarrel_wheel_station->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchOccupancyBarrel_wheel->getTH1()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchOccupancyBarrel_wheel_station->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=9; ++i )
  {
    TString binLabel = Form("Disk %d", i-5);
    h_refPunchOccupancyEndcap_disk  ->getTH1()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refPunchOccupancyEndcap_disk_ring  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchOccupancyEndcap_disk  ->getTH1()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchOccupancyEndcap_disk_ring  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=4; ++i )
  {
    TString binLabel = Form("Station %d", i);
    h_refPunchOccupancyBarrel_station  ->getTH1()->GetXaxis()->SetBinLabel(i, binLabel);
    h_refPunchOccupancyBarrel_wheel_station  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    h_recPunchOccupancyBarrel_station  ->getTH1()->GetXaxis()->SetBinLabel(i, binLabel);
    h_recPunchOccupancyBarrel_wheel_station  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=4; ++i )
  {
    TString binLabel = Form("Ring %d", i);
    h_refPunchOccupancyEndcap_disk_ring  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    h_recPunchOccupancyEndcap_disk_ring  ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
  }
}

RPCRecHitValid::~RPCRecHitValid()
{
}

void RPCRecHitValid::beginJob()
{
}

void RPCRecHitValid::endJob()
{
}

void RPCRecHitValid::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup)
{
  if ( !dbe_ ) return;
  h_eventCount->Fill(3);

  // Book roll-by-roll histograms
  edm::ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);

  int nRPCRollBarrel = 0, nRPCRollEndcap = 0;

  TrackingGeometry::DetContainer rpcDets = rpcGeom->dets();
  for ( auto det : rpcDets )
  {
    RPCChamber* rpcCh = dynamic_cast<RPCChamber*>(det);
    if ( !rpcCh ) continue;

    std::vector<const RPCRoll*> rolls = rpcCh->rolls();
    for ( auto roll : rolls )
    {
      if ( !roll ) continue;

      //RPCGeomServ rpcSrv(roll->id());
      const int rawId = roll->geographicalId().rawId();
      //if ( !roll->specs()->isRPC() ) { cout << "\nNoRPC : " << rpcSrv.name() << ' ' << rawId << endl; continue; }

      if ( roll->isBarrel() )
      {
        detIdToIndexMapBarrel_[rawId] = nRPCRollBarrel;
        //rollIdToNameMapBarrel_[rawId] = rpcSrv.name();
        ++nRPCRollBarrel;
      }
      else
      {
        detIdToIndexMapEndcap_[rawId] = nRPCRollEndcap;
        //rollIdToNameMapEndcap_[rawId] = rpcSrv.name();
        ++nRPCRollEndcap;
      }
    }
  }

  dbe_->setCurrentFolder(subDir_+"/Occupancy");
  h_matchOccupancyBarrel_detId = dbe_->book1D("MatchOccupancyBarrel_detId", "Matched hit occupancy;roll index (can be arbitrary)", nRPCRollBarrel, 0, nRPCRollBarrel);
  h_matchOccupancyEndcap_detId = dbe_->book1D("MatchOccupancyEndcap_detId", "Matched hit occupancy;roll index (can be arbitrary)", nRPCRollEndcap, 0, nRPCRollEndcap);
  h_refOccupancyBarrel_detId = dbe_->book1D("RefOccupancyBarrel_detId", "Reference hit occupancy;roll index (can be arbitrary)", nRPCRollBarrel, 0, nRPCRollBarrel);
  h_refOccupancyEndcap_detId = dbe_->book1D("RefOccupancyEndcap_detId", "Reference hit occupancy;roll index (can be arbitrary)", nRPCRollEndcap, 0, nRPCRollEndcap);
  h_noiseOccupancyBarrel_detId = dbe_->book1D("NoiseOccupancyBarrel_detId", "Noise occupancy;roll index (can be arbitrary)", nRPCRollBarrel, 0, nRPCRollBarrel);
  h_noiseOccupancyEndcap_detId = dbe_->book1D("NoiseOccupancyEndcap_detId", "Noise occupancy;roll index (can be arbitrary)", nRPCRollEndcap, 0, nRPCRollEndcap);

  h_matchOccupancyBarrel_detId->getTH1()->SetMinimum(0);
  h_matchOccupancyEndcap_detId->getTH1()->SetMinimum(0);
  h_refOccupancyBarrel_detId  ->getTH1()->SetMinimum(0);
  h_refOccupancyEndcap_detId  ->getTH1()->SetMinimum(0);
  h_noiseOccupancyBarrel_detId->getTH1()->SetMinimum(0);
  h_noiseOccupancyEndcap_detId->getTH1()->SetMinimum(0);

  h_rollAreaBarrel_detId = dbe_->bookProfile("RollAreaBarrel_detId", "Roll area;roll index;Area", nRPCRollBarrel, 0., 1.*nRPCRollBarrel, 0., 1e5);
  h_rollAreaEndcap_detId = dbe_->bookProfile("RollAreaEndcap_detId", "Roll area;roll index;Area", nRPCRollEndcap, 0., 1.*nRPCRollEndcap, 0., 1e5);

  for ( auto detIdToIndex : detIdToIndexMapBarrel_ )
  {
    const int rawId = detIdToIndex.first;
    const int index = detIdToIndex.second;

    const RPCDetId rpcDetId = static_cast<const RPCDetId>(rawId);
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(rpcDetId));

    //RPCGeomServ rpcSrv(roll->id());
    //if ( !roll->specs()->isRPC() ) { cout << "\nNoRPC : " << rpcSrv.name() << ' ' << rawId << endl; continue; }

    const StripTopology& topol = roll->specificTopology();
    const double area = topol.stripLength()*topol.nstrips()*topol.pitch();

    h_rollAreaBarrel_detId->Fill(index, area);
  }

  for ( auto detIdToIndex : detIdToIndexMapEndcap_ )
  {
    const int rawId = detIdToIndex.first;
    const int index = detIdToIndex.second;

    const RPCDetId rpcDetId = static_cast<const RPCDetId>(rawId);
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(rpcDetId));

    //RPCGeomServ rpcSrv(roll->id());
    //if ( !roll->specs()->isRPC() ) { cout << "\nNoRPC : " << rpcSrv.name() << ' ' << rawId << endl; continue; }

    const StripTopology& topol = roll->specificTopology();
    const double area = topol.stripLength()*topol.nstrips()*topol.pitch();

    h_rollAreaEndcap_detId->Fill(index, area);
  }
}

void RPCRecHitValid::endRun(const edm::Run& run, const edm::EventSetup& eventSetup)
{
  if ( !dbe_ ) return;

}

void RPCRecHitValid::analyze(const edm::Event& event, const edm::EventSetup& eventSetup)
{
  if ( !dbe_ ) return;
  h_eventCount->Fill(1);

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

  // Get SimParticles
  edm::Handle<TrackingParticleCollection> simParticleHandle;
  if ( !event.getByLabel(simParticleLabel_, simParticleHandle) )
  {
    edm::LogInfo("RPCRecHitValid") << "Cannot find TrackingParticle collection\n";
    return;
  }

  typedef std::pair<TrackingParticleRef, TrackPSimHitRef> SimHitTPPair;
  typedef std::vector<SimHitTPPair> SimHitTPAssociationList;
  // Get SimParticle to SimHit association map
  edm::Handle<SimHitTPAssociationList> simHitsTPAssoc;
  if ( !event.getByLabel(simHitAssocLabel_, simHitsTPAssoc) )
  {
    edm::LogInfo("RPCRecHitValid") << "Cannot find TrackingParticle to SimHit association map\n";
    return;
  }

  // Get RecoMuons
  edm::Handle<edm::View<reco::Muon> > muonHandle;
  if ( !event.getByLabel(muonLabel_, muonHandle) )
  {
    edm::LogInfo("RPCRecHitValid") << "Cannot find muon collection\n";
    return;
  }

  typedef edm::PSimHitContainer::const_iterator SimHitIter;
  typedef RPCRecHitCollection::const_iterator RecHitIter;
  typedef std::vector<TrackPSimHitRef> SimHitRefs;

  // TrackingParticles with (and without) RPC simHits
  SimHitRefs muonSimHits, pthrSimHits;

  for ( int i=0, n=simParticleHandle->size(); i<n; ++i )
  {
    TrackingParticleRef simParticle(simParticleHandle, i);
    if ( simParticle->pt() < 1.0 or simParticle->p() < 2.5 ) continue; // globalMuon acceptance

    // Collect SimHits from this Tracking Particle
    SimHitRefs simHitsFromParticle;
    auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(),
                                  std::make_pair(simParticle, TrackPSimHitRef()),
                                  SimHitTPAssociationProducer::simHitTPAssociationListGreater);
    for ( auto simParticleToHit = range.first; simParticleToHit != range.second; ++simParticleToHit )
    {
      auto simHit = simParticleToHit->second;
      const DetId detId(simHit->detUnitId());
      if ( detId.det() != DetId::Muon or detId.subdetId() != MuonSubdetId::RPC ) continue;

      simHitsFromParticle.push_back(simParticleToHit->second);
    }
    const int nRPCHit = simHitsFromParticle.size();
    const bool hasRPCHit = nRPCHit > 0;

    if ( abs(simParticle->pdgId()) == 13 )
    {
      muonSimHits.insert(muonSimHits.end(), simHitsFromParticle.begin(), simHitsFromParticle.end());

      // Count number of Barrel hits and Endcap hits
      int nRPCHitBarrel = 0;
      int nRPCHitEndcap = 0;
      for ( auto simHit : simHitsFromParticle )
      {
        const RPCDetId rpcDetId = static_cast<const RPCDetId>(simHit->detUnitId());
        const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(rpcDetId));
        if ( !roll ) continue;

        if ( rpcDetId.region() == 0 ) ++nRPCHitBarrel;
        else ++nRPCHitEndcap;
      }

      // Fill TrackingParticle related histograms
      h_nRPCHitPerSimMuon->Fill(nRPCHit);
      if ( nRPCHitBarrel and nRPCHitEndcap )
      {
        h_nRPCHitPerSimMuonOverlap->Fill(nRPCHit);
        h_simMuonOverlap_pt->Fill(simParticle->pt());
        h_simMuonOverlap_eta->Fill(simParticle->eta());
        h_simMuonOverlap_phi->Fill(simParticle->phi());
      }
      else if ( nRPCHitBarrel )
      {
        h_nRPCHitPerSimMuonBarrel->Fill(nRPCHit);
        h_simMuonBarrel_pt->Fill(simParticle->pt());
        h_simMuonBarrel_eta->Fill(simParticle->eta());
        h_simMuonBarrel_phi->Fill(simParticle->phi());
      }
      else if ( nRPCHitEndcap )
      {
        h_nRPCHitPerSimMuonEndcap->Fill(nRPCHit);
        h_simMuonEndcap_pt->Fill(simParticle->pt());
        h_simMuonEndcap_eta->Fill(simParticle->eta());
        h_simMuonEndcap_phi->Fill(simParticle->phi());
      }
      else
      {
        h_simMuonNoRPC_pt->Fill(simParticle->pt());
        h_simMuonNoRPC_eta->Fill(simParticle->eta());
        h_simMuonNoRPC_phi->Fill(simParticle->phi());
      }
    }
    else
    {
      pthrSimHits.insert(pthrSimHits.end(), simHitsFromParticle.begin(), simHitsFromParticle.end());
    }

    if ( hasRPCHit )
    {
      switch ( simParticle->pdgId() )
      {
        case    13: h_simParticleType->Fill( 0); break;
        case   -13: h_simParticleType->Fill( 1); break;
        case    11: h_simParticleType->Fill( 2); break;
        case   -11: h_simParticleType->Fill( 3); break;
        case   211: h_simParticleType->Fill( 4); break;
        case  -211: h_simParticleType->Fill( 5); break;
        case   321: h_simParticleType->Fill( 6); break;
        case  -321: h_simParticleType->Fill( 7); break;
        case  2212: h_simParticleType->Fill( 8); break;
        case -2212: h_simParticleType->Fill( 9); break;
        default:    h_simParticleType->Fill(10); break;
      }
    }
  }

  // Loop over muon simHits, fill histograms which does not need associations
  int nRefHitBarrel = 0, nRefHitEndcap = 0;
  for ( auto simHit : muonSimHits )
  {
    const RPCDetId detId = static_cast<const RPCDetId>(simHit->detUnitId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId));

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subSector = roll->id().subsector();

    if ( region == 0 )
    {
      ++nRefHitBarrel;
      h_.refHitOccupancyBarrel_wheel->Fill(ring);
      h_.refHitOccupancyBarrel_station->Fill(station);
      h_.refHitOccupancyBarrel_wheel_station->Fill(ring, station);

      h_refOccupancyBarrel_detId->Fill(detIdToIndexMapBarrel_[simHit->detUnitId()]);
    }
    else
    {
      ++nRefHitEndcap;
      h_.refHitOccupancyEndcap_disk->Fill(region*station);
      h_.refHitOccupancyEndcap_disk_ring->Fill(region*station, ring);

      h_refOccupancyEndcap_detId->Fill(detIdToIndexMapEndcap_[simHit->detUnitId()]);
    }
  }

  // Loop over punch-through simHits, fill histograms which does not need associations
  for ( auto simHit : pthrSimHits )
  {
    const RPCDetId detId = static_cast<const RPCDetId>(simHit->detUnitId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId()));

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subSector = roll->id().subsector();

    if ( region == 0 )
    {
      ++nRefHitBarrel;
      h_refPunchOccupancyBarrel_wheel->Fill(ring);
      h_refPunchOccupancyBarrel_station->Fill(station);
      h_refPunchOccupancyBarrel_wheel_station->Fill(ring, station);

      h_refOccupancyBarrel_detId->Fill(detIdToIndexMapBarrel_[simHit->detUnitId()]);
    }
    else
    {
      ++nRefHitEndcap;
      h_refPunchOccupancyEndcap_disk->Fill(region*station);
      h_refPunchOccupancyEndcap_disk_ring->Fill(region*station, ring);

      h_refOccupancyEndcap_detId->Fill(detIdToIndexMapEndcap_[simHit->detUnitId()]);
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
      h_.recHitOccupancyEndcap_disk_ring->Fill(region*station, ring);
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
  typedef std::map<TrackPSimHitRef, RecHitIter> SimToRecHitMap;
  SimToRecHitMap simToRecHitMap;

  for ( auto simHit : muonSimHits )
  {
    const RPCDetId simDetId = static_cast<const RPCDetId>(simHit->detUnitId());
    //const RPCRoll* simRoll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(simDetId));

    const double simX = simHit->localPosition().x();

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
      SimToRecHitMap::const_iterator prevSimToReco = simToRecHitMap.find(simHit);
      if ( prevSimToReco == simToRecHitMap.end() )
      {
        simToRecHitMap.insert(std::make_pair(simHit, recHitIter));
      }
      else
      {
        const double oldDx = fabs(prevSimToReco->second->localPosition().x() - simX);

        if ( newDx < oldDx )
        {
          simToRecHitMap[simHit] = recHitIter;
        }
      }
    }
  }

  // Now we have simHit-recHit mapping
  // So we can fill up relavant histograms
  int nMatchHitBarrel = 0, nMatchHitEndcap = 0;
  for ( auto match : simToRecHitMap )
  {
    TrackPSimHitRef simHit = match.first;
    RecHitIter recHitIter = match.second;

    const RPCDetId detId = static_cast<const RPCDetId>(simHit->detUnitId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(detId));

    const int region = roll->id().region();
    const int ring = roll->id().ring();
    //const int sector = roll->id().sector();
    const int station = roll->id().station();
    //const int layer = roll->id().layer();
    //const int subsector = roll->id().subsector();

    const double simX = simHit->localPosition().x();
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
      h_.matchOccupancyBarrel_wheel->Fill(ring);
      h_.matchOccupancyBarrel_station->Fill(station);
      h_.matchOccupancyBarrel_wheel_station->Fill(ring, station);

      h_.res_wheel_res->Fill(ring, dX);
      h_.res_station_res->Fill(station, dX);
      h_.pull_wheel_pull->Fill(ring, pull);
      h_.pull_station_pull->Fill(station, pull);

      h_matchOccupancyBarrel_detId->Fill(detIdToIndexMapBarrel_[detId.rawId()]);
    }
    else
    {
      ++nMatchHitEndcap;
      h_.resEndcap->Fill(dX);
      h_.pullEndcap->Fill(pull);
      h_.matchOccupancyEndcap_disk->Fill(region*station);
      h_.matchOccupancyEndcap_disk_ring->Fill(region*station, ring);

      h_.res_disk_res->Fill(region*station, dX);
      h_.res_ring_res->Fill(ring, dX);
      h_.pull_disk_pull->Fill(region*station, pull);
      h_.pull_ring_pull->Fill(ring, pull);

      h_matchOccupancyEndcap_detId->Fill(detIdToIndexMapEndcap_[detId.rawId()]);
    }

  }
  h_.nMatchHitBarrel->Fill(nMatchHitBarrel);
  h_.nMatchHitEndcap->Fill(nMatchHitEndcap);

  // Reco Muon hits
  for ( edm::View<reco::Muon>::const_iterator muon = muonHandle->begin();
        muon != muonHandle->end(); ++muon )
  {
    if ( !muon->isGlobalMuon() ) continue;

    int nRPCHitBarrel = 0;
    int nRPCHitEndcap = 0;

    const reco::TrackRef glbTrack = muon->globalTrack();
    for ( trackingRecHit_iterator recHit = glbTrack->recHitsBegin();
          recHit != glbTrack->recHitsEnd(); ++recHit )
    {
      if ( !(*recHit)->isValid() ) continue;
      const DetId detId = (*recHit)->geographicalId();
      if ( detId.det() != DetId::Muon or detId.subdetId() != MuonSubdetId::RPC ) continue;
      const RPCDetId rpcDetId = static_cast<const RPCDetId>(detId);

      if ( rpcDetId.region() == 0 ) ++nRPCHitBarrel;
      else ++nRPCHitEndcap;
    }

    const int nRPCHit = nRPCHitBarrel + nRPCHitEndcap;
    h_nRPCHitPerRecoMuon->Fill(nRPCHit);
    if ( nRPCHitBarrel and nRPCHitEndcap )
    {
      h_nRPCHitPerRecoMuonOverlap->Fill(nRPCHit);
      h_recoMuonOverlap_pt->Fill(muon->pt());
      h_recoMuonOverlap_eta->Fill(muon->eta());
      h_recoMuonOverlap_phi->Fill(muon->phi());
    }
    else if ( nRPCHitBarrel )
    {
      h_nRPCHitPerRecoMuonBarrel->Fill(nRPCHit);
      h_recoMuonBarrel_pt->Fill(muon->pt());
      h_recoMuonBarrel_eta->Fill(muon->eta());
      h_recoMuonBarrel_phi->Fill(muon->phi());
    }
    else if ( nRPCHitEndcap )
    {
      h_nRPCHitPerRecoMuonEndcap->Fill(nRPCHit);
      h_recoMuonEndcap_pt->Fill(muon->pt());
      h_recoMuonEndcap_eta->Fill(muon->eta());
      h_recoMuonEndcap_phi->Fill(muon->phi());
    }
    else
    {
      h_recoMuonNoRPC_pt->Fill(muon->pt());
      h_recoMuonNoRPC_eta->Fill(muon->eta());
      h_recoMuonNoRPC_phi->Fill(muon->phi());
    }
  }

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
    for ( auto match : simToRecHitMap )
    {
      if ( recHitIter == match.second )
      {
        matched = true;
        break;
      }
    }

    if ( !matched )
    {
      // FIXME : kept for backward compatibility //
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
      // End of FIXME //

      int nPunchMatched = 0;
      // Check if this recHit came from non-muon simHit
      for ( auto simHit : pthrSimHits )
      {
        const int absSimHitPType = abs(simHit->particleType());
        if ( absSimHitPType == 13 ) continue;

        const RPCDetId simDetId = static_cast<const RPCDetId>(simHit->detUnitId());
        if ( simDetId == detId ) ++nPunchMatched;
      }

      if ( nPunchMatched > 0 )
      {
        if ( region == 0 )
        {
          h_recPunchOccupancyBarrel_wheel->Fill(ring);
          h_recPunchOccupancyBarrel_station->Fill(station);
          h_recPunchOccupancyBarrel_wheel_station->Fill(ring, station);
        }
        else
        {
          h_recPunchOccupancyEndcap_disk->Fill(region*station);
          h_recPunchOccupancyEndcap_disk_ring->Fill(region*station, ring);
        }
      }
    }
  }

  // Find noise recHits : RecHits without SimHit match
  for ( RecHitIter recHitIter = recHitHandle->begin();
        recHitIter != recHitHandle->end(); ++recHitIter )
  {
    const RPCDetId recDetId = static_cast<const RPCDetId>(recHitIter->rpcId());
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(recDetId));

    const int region = roll->id().region();
    //    const int ring = roll->id().ring(); // UNUSED VARIABLE
    //const int sector = roll->id().sector();
    //    const int station = roll->id().station(); // UNUSED VARIABLE
    //const int layer = roll->id().layer();
    //const int subsector = roll->id().subsector();

    const double recX = recHitIter->localPosition().x();
    const double recErrX = sqrt(recHitIter->localPositionError().xx());

    bool matched = false;
    for ( SimHitIter simHitIter = simHitHandle->begin();
          simHitIter != simHitHandle->end(); ++simHitIter )
    {
      const RPCDetId simDetId = static_cast<const RPCDetId>(simHitIter->detUnitId());
      const RPCRoll* simRoll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(simDetId));
      if ( !simRoll ) continue;

      if ( simDetId != recDetId ) continue;

      const double simX = simHitIter->localPosition().x();
      const double dX = fabs(recX-simX);

      if ( dX/recErrX < 5 )
      {
        matched = true;
        break;
      }
    }

    if ( !matched )
    {
      if ( region == 0 )
      {
        h_noiseOccupancyBarrel_detId->Fill(detIdToIndexMapBarrel_[recDetId.rawId()]);
      }
      else
      {
        h_noiseOccupancyEndcap_detId->Fill(detIdToIndexMapEndcap_[recDetId.rawId()]);
      }
    }
  }

  h_eventCount->Fill(2);
}

DEFINE_FWK_MODULE(RPCRecHitValid);

