// system include files
#include <memory>
#include <fstream>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include "DataFormats/Provenance/interface/Timestamp.h"

#include <DataFormats/MuonDetId/interface/GEMDetId.h>

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "RecoMuon/DetLayers/interface/MuRodBarrelLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include "RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRing.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

//Structures used for filling ME
#include "Validation/MuonGEMRecHits/src/SimRecStructures.h"

// using namespace std;
// using namespace edm;

class MuonGEMRecHits : public edm::EDAnalyzer 
{
public:
  explicit MuonGEMRecHits(const edm::ParameterSet&);
  ~MuonGEMRecHits();
    
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
private:
    
    bool isGEMRecHitMatched(const MyGEMRecHit& gem_recHit_, const MyGEMSimHit& gem_sh);
    bool isSimTrackGood(const SimTrack &t);
    void bookHistograms();
    
    virtual void beginJob() override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    
    bool debug_;
    edm::InputTag gemRecHitInput_;
    edm::InputTag gemSimHitInput_;
    edm::InputTag simTrackInput_;
    std::string folderPath_;
    std::string outputFile_;
    
    DQMStore * dbe_;
    std::map<std::string, MonitorElement*> meCollection_;
    
    edm::Handle<GEMRecHitCollection> gemRecHits_;
    edm::Handle<edm::PSimHitContainer> gemSimHits_;
    edm::Handle<edm::SimTrackContainer> simTracks_;
    edm::Handle<edm::SimVertexContainer> simVertices_;
    edm::ESHandle<GEMGeometry> gem_geom_;
    
    const GEMGeometry* gem_geometry_;
    bool hasGEMGeometry_;

    /*-------------------------------------------------*/
    //These structures are defined in "SimRecStructures.h"
    MyGEMRecHit gem_recHit_;
    MyGEMSimHit gem_simHit_;
    MySimTrack track_;
    /*-------------------------------------------------*/
};


MuonGEMRecHits::MuonGEMRecHits(const edm::ParameterSet& iConfig) : 
  debug_(iConfig.getUntrackedParameter<bool>("debug")),
  gemRecHitInput_(iConfig.getUntrackedParameter<edm::InputTag>("gemRecHitInput")),
  gemSimHitInput_(iConfig.getUntrackedParameter<edm::InputTag>("gemSimHitInput")),
  simTrackInput_(iConfig.getUntrackedParameter<edm::InputTag>("simTrackInput")),
  folderPath_(iConfig.getUntrackedParameter<std::string>("folderPath")),
  outputFile_(iConfig.getUntrackedParameter<std::string>("outputFile"))
{
  hasGEMGeometry_ = false;

  dbe_ = edm::Service<DQMStore>().operator->();
    
}


MuonGEMRecHits::~MuonGEMRecHits()
{
}


bool MuonGEMRecHits::isGEMRecHitMatched(const MyGEMRecHit& gem_recHit_, const MyGEMSimHit& gem_sh)
{
  Int_t gem_region = gem_recHit_.region;
  Int_t gem_layer = gem_recHit_.layer;
  Int_t gem_station = gem_recHit_.station;
  Int_t gem_chamber = gem_recHit_.chamber;
  Int_t gem_roll = gem_recHit_.roll;
  Int_t gem_firstStrip = gem_recHit_.firstClusterStrip;
  Int_t gem_cls = gem_recHit_.clusterSize;
    
  Int_t gem_sh_region = gem_sh.region;
  Int_t gem_sh_layer = gem_sh.layer;
  Int_t gem_sh_station = gem_sh.station;
  Int_t gem_sh_chamber = gem_sh.chamber;
  Int_t gem_sh_roll = gem_sh.roll;
  Int_t gem_sh_strip = gem_sh.strip;
    
  std::vector<int> stripsFired;
  for(int i = gem_firstStrip; i < (gem_firstStrip + gem_cls); i++){
    stripsFired.push_back(i);
  }
    
  const bool cond1(gem_sh_region == gem_region and gem_sh_layer == gem_layer and gem_sh_station == gem_station);
  const bool cond2(gem_sh_chamber == gem_chamber and gem_sh_roll == gem_roll);
  const bool cond3(std::find(stripsFired.begin(), stripsFired.end(), (gem_sh_strip + 1)) != stripsFired.end());

  return (cond1 and cond2 and cond3);
}


void
MuonGEMRecHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if (not hasGEMGeometry_) return;
  
  iEvent.getByLabel(gemRecHitInput_, gemRecHits_);
  iEvent.getByLabel(gemSimHitInput_, gemSimHits_);
  iEvent.getByLabel(simTrackInput_, simTracks_);

  std::vector<int> trackIds;
  std::vector<int> trackType;

  const edm::SimTrackContainer & sim_trks = *simTracks_.product();
  
  for (auto& t: sim_trks) {
    if (!isSimTrackGood(t)) continue;
    trackType.push_back(t.type());
    trackIds.push_back(t.trackId());
  }

  for (edm::PSimHitContainer::const_iterator itHit = gemSimHits_->begin(); itHit!=gemSimHits_->end(); ++itHit) {
        
    if(abs(itHit->particleType()) != 13) continue;
    if(std::find(trackIds.begin(), trackIds.end(), itHit->trackId()) == trackIds.end()) continue;
    
    gem_simHit_.eventNumber = iEvent.id().event();
    gem_simHit_.detUnitId = itHit->detUnitId();
    gem_simHit_.particleType = itHit->particleType();
    gem_simHit_.x = itHit->localPosition().x();
    gem_simHit_.y = itHit->localPosition().y();
    gem_simHit_.energyLoss = itHit->energyLoss();
    gem_simHit_.pabs = itHit->pabs();
    gem_simHit_.timeOfFlight = itHit->timeOfFlight();
    
    const GEMDetId id(itHit->detUnitId());
    
    gem_simHit_.region = id.region();
    gem_simHit_.ring = id.ring();
    gem_simHit_.station = id.station();
    gem_simHit_.layer = id.layer();
    gem_simHit_.chamber = id.chamber();
    gem_simHit_.roll = id.roll();
    
    const LocalPoint p0(0., 0., 0.);
    const GlobalPoint Gp0(gem_geometry_->idToDet(itHit->detUnitId())->surface().toGlobal(p0));
    
    gem_simHit_.Phi_0 = Gp0.phi();
    gem_simHit_.R_0 = Gp0.perp();
    gem_simHit_.DeltaPhi = atan(-1*id.region()*pow(-1,id.chamber())*itHit->localPosition().x()/(Gp0.perp() + itHit->localPosition().y()));
    
    const LocalPoint hitLP(itHit->localPosition());
    const GlobalPoint hitGP(gem_geometry_->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP));
    gem_simHit_.globalR = hitGP.perp();
    gem_simHit_.globalEta = hitGP.eta();
    gem_simHit_.globalPhi = hitGP.phi();
    gem_simHit_.globalX = hitGP.x();
    gem_simHit_.globalY = hitGP.y();
    gem_simHit_.globalZ = hitGP.z();
    
    //  Now filling strip info using entry point rather than local position to be
    //  consistent with digi strips. To change back, just switch the comments - WHF
    //  gem_simHit_.strip=gem_geometry_->etaPartition(itHit->detUnitId())->strip(hitLP);
    const LocalPoint hitEP(itHit->entryPoint());
    gem_simHit_.strip = gem_geometry_->etaPartition(itHit->detUnitId())->strip(hitEP);
    
    int count = 0;
    //std::cout<<"SimHit: region "<<gem_simHit_.region<<" station "<<gem_simHit_.station<<" layer "<<gem_simHit_.layer<<" chamber "<<gem_simHit_.chamber<<" roll "<<gem_simHit_.roll<<" strip "<<gem_simHit_.strip<<" type "<<itHit->particleType()<<" id "<<itHit->trackId()<<std::endl;

    for (GEMRecHitCollection::const_iterator recHit = gemRecHits_->begin(); recHit != gemRecHits_->end(); ++recHit) {
      gem_recHit_.x = recHit->localPosition().x();
      gem_recHit_.xErr = recHit->localPositionError().xx();
      gem_recHit_.y = recHit->localPosition().y();
      gem_recHit_.detId = (Short_t) (*recHit).gemId();
      gem_recHit_.bx = recHit->BunchX();
      gem_recHit_.clusterSize = recHit->clusterSize();
      gem_recHit_.firstClusterStrip = recHit->firstClusterStrip();

      GEMDetId id((*recHit).gemId());
      
      gem_recHit_.region = (Short_t) id.region();
      gem_recHit_.ring = (Short_t) id.ring();
      gem_recHit_.station = (Short_t) id.station();
      gem_recHit_.layer = (Short_t) id.layer();
      gem_recHit_.chamber = (Short_t) id.chamber();
      gem_recHit_.roll = (Short_t) id.roll();

      LocalPoint hitLP = recHit->localPosition();
      GlobalPoint hitGP = gem_geometry_->idToDet((*recHit).gemId())->surface().toGlobal(hitLP);

      gem_recHit_.globalR = hitGP.perp();
      gem_recHit_.globalEta = hitGP.eta();
      gem_recHit_.globalPhi = hitGP.phi();
      gem_recHit_.globalX = hitGP.x();
      gem_recHit_.globalY = hitGP.y();
      gem_recHit_.globalZ = hitGP.z();
      
      gem_recHit_.x_sim = gem_simHit_.x;
      gem_recHit_.y_sim = gem_simHit_.y;
      gem_recHit_.globalEta_sim = gem_simHit_.globalEta;
      gem_recHit_.globalPhi_sim = gem_simHit_.globalPhi;
      gem_recHit_.globalX_sim = gem_simHit_.globalX;
      gem_recHit_.globalY_sim = gem_simHit_.globalY;
      gem_recHit_.globalZ_sim = gem_simHit_.globalZ;
      gem_recHit_.pull = (gem_simHit_.x - gem_recHit_.x) / gem_recHit_.xErr;

      // abbreviations
      int re(gem_recHit_.region);
      int st(gem_recHit_.station);
      int la(gem_recHit_.layer);

    
      /*-----------BunchCrossing----------------*/
      meCollection_["bxDistribution"]->Fill(gem_recHit_.bx);
      if(st==1) meCollection_["bxDistribution_st1"]->Fill(gem_recHit_.bx);
      if(st==2) meCollection_["bxDistribution_st2"]->Fill(gem_recHit_.bx);
      if(st==3) meCollection_["bxDistribution_st3"]->Fill(gem_recHit_.bx);

      if(gem_recHit_.bx != 0) continue;
      if(isGEMRecHitMatched(gem_recHit_, gem_simHit_)) {
	if (debug_)
	  std::cout<<"RecHit: region "<<re<<" station "<<st
		   <<" layer "<<la<<" chamber "<<gem_recHit_.chamber
		   <<" roll "<<gem_recHit_.roll<<" firstStrip "<<gem_recHit_.firstClusterStrip
		   <<" cls "<<gem_recHit_.clusterSize<<" bx "<<gem_recHit_.bx<<std::endl;
	
        /*----------------ClustersSize--------------------------*/
	meCollection_["clsDistribution"]->Fill(gem_recHit_.clusterSize);

	if(re==-1 and la==1) meCollection_["clsDistribution_rm1_l1"]->Fill(gem_recHit_.clusterSize);
	if(re==-1 and la==2) meCollection_["clsDistribution_rm1_l2"]->Fill(gem_recHit_.clusterSize);
	if(re== 1 and la==1) meCollection_["clsDistribution_rp1_l1"]->Fill(gem_recHit_.clusterSize);
	if(re== 1 and la==2) meCollection_["clsDistribution_rp1_l2"]->Fill(gem_recHit_.clusterSize);

        /*-----------------------X pull--------------------------*/
	meCollection_["recHitPullX"]->Fill(gem_recHit_.pull);
        
	if(st==1 and re==-1 and la==1) meCollection_["recHitPullX_rm1_st1_l1"]->Fill(gem_recHit_.pull);
	if(st==1 and re==-1 and la==2) meCollection_["recHitPullX_rm1_st1_l2"]->Fill(gem_recHit_.pull);
	if(st==1 and re== 1 and la==1) meCollection_["recHitPullX_rp1_st1_l1"]->Fill(gem_recHit_.pull);
	if(st==1 and re== 1 and la==2) meCollection_["recHitPullX_rp1_st1_l2"]->Fill(gem_recHit_.pull);
	if(st==2 and re==-1 and la==1) meCollection_["recHitPullX_rm1_st2_l1"]->Fill(gem_recHit_.pull);
	if(st==2 and re==-1 and la==2) meCollection_["recHitPullX_rm1_st2_l2"]->Fill(gem_recHit_.pull);
	if(st==2 and re== 1 and la==1) meCollection_["recHitPullX_rp1_st2_l1"]->Fill(gem_recHit_.pull);
	if(st==2 and re== 1 and la==2) meCollection_["recHitPullX_rp1_st2_l2"]->Fill(gem_recHit_.pull);
	if(st==3 and re==-1 and la==1) meCollection_["recHitPullX_rm1_st3_l1"]->Fill(gem_recHit_.pull);
	if(st==3 and re==-1 and la==2) meCollection_["recHitPullX_rm1_st3_l2"]->Fill(gem_recHit_.pull);
	if(st==3 and re== 1 and la==1) meCollection_["recHitPullX_rp1_st3_l1"]->Fill(gem_recHit_.pull);
	if(st==3 and re== 1 and la==2) meCollection_["recHitPullX_rp1_st3_l2"]->Fill(gem_recHit_.pull);

        /*---------------------DeltaPhi-------------------------------*/
	const double deltaPhi(gem_recHit_.globalPhi - gem_simHit_.globalPhi);
        
	meCollection_["recHitDPhi"]->Fill(deltaPhi);
	if(st==1) meCollection_["recHitDPhi_st1"]->Fill(deltaPhi);
	if(st==2) meCollection_["recHitDPhi_st2"]->Fill(deltaPhi);
	if(st==3) meCollection_["recHitDPhi_st3"]->Fill(deltaPhi);
        
	if(st==1 and gem_recHit_.clusterSize==1)meCollection_["recHitDPhi_st1_cls1"]->Fill(deltaPhi);
	if(st==2 and gem_recHit_.clusterSize==1)meCollection_["recHitDPhi_st2_cls1"]->Fill(deltaPhi);
	if(st==3 and gem_recHit_.clusterSize==1)meCollection_["recHitDPhi_st3_cls1"]->Fill(deltaPhi);
	if(st==1 and gem_recHit_.clusterSize==2)meCollection_["recHitDPhi_st1_cls2"]->Fill(deltaPhi);
	if(st==2 and gem_recHit_.clusterSize==2)meCollection_["recHitDPhi_st2_cls2"]->Fill(deltaPhi);
	if(st==3 and gem_recHit_.clusterSize==2)meCollection_["recHitDPhi_st3_cls2"]->Fill(deltaPhi);
	if(st==1 and gem_recHit_.clusterSize==3)meCollection_["recHitDPhi_st1_cls3"]->Fill(deltaPhi);
	if(st==2 and gem_recHit_.clusterSize==3)meCollection_["recHitDPhi_st2_cls3"]->Fill(deltaPhi);
	if(st==3 and gem_recHit_.clusterSize==3)meCollection_["recHitDPhi_st3_cls3"]->Fill(deltaPhi);
        
        /*---------------------Occupancy XY---------------------------*/
	if(st==1 and re==-1 and la==1) meCollection_["localrh_xy_rm1_st1_l1"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==1 and re==-1 and la==2) meCollection_["localrh_xy_rm1_st1_l2"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==1 and re== 1 and la==1) meCollection_["localrh_xy_rp1_st1_l1"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==1 and re== 1 and la==2) meCollection_["localrh_xy_rp1_st1_l2"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==2 and re==-1 and la==1) meCollection_["localrh_xy_rm1_st2_l1"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==2 and re==-1 and la==2) meCollection_["localrh_xy_rm1_st2_l2"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==2 and re== 1 and la==1) meCollection_["localrh_xy_rp1_st2_l1"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==2 and re== 1 and la==2) meCollection_["localrh_xy_rp1_st2_l2"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==3 and re==-1 and la==1) meCollection_["localrh_xy_rm1_st3_l1"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==3 and re==-1 and la==2) meCollection_["localrh_xy_rm1_st3_l2"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==3 and re== 1 and la==1) meCollection_["localrh_xy_rp1_st3_l1"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	if(st==3 and re== 1 and la==2) meCollection_["localrh_xy_rp1_st3_l2"]->Fill(gem_recHit_.globalX,gem_recHit_.globalY);
	
	const double glb_R(sqrt(gem_recHit_.globalX*gem_recHit_.globalX+gem_recHit_.globalY*gem_recHit_.globalY));
	if(st==1 and re==-1) meCollection_["localrh_zr_rm1_st1"]->Fill(gem_recHit_.globalZ,glb_R);
	if(st==1 and re==1) meCollection_["localrh_zr_rp1_st1"]->Fill(gem_recHit_.globalZ,glb_R);
        
        /*-------------------Strips--------------------------------------*/
        
	meCollection_["strip_rh_tot"]->Fill(gem_recHit_.firstClusterStrip);
        
	if(st==1 and re==-1 and la==1) meCollection_["strip_rh_rm1_st1_l1_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==1 and re==-1 and la==2) meCollection_["strip_rh_rm1_st1_l2_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==1 and re== 1 and la==1) meCollection_["strip_rh_rp1_st1_l1_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==1 and re== 1 and la==2) meCollection_["strip_rh_rp1_st1_l2_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==2 and re==-1 and la==1) meCollection_["strip_rh_rm1_st2_l1_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==2 and re==-1 and la==2) meCollection_["strip_rh_rm1_st2_l2_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==2 and re== 1 and la==1) meCollection_["strip_rh_rp1_st2_l1_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==2 and re== 1 and la==2) meCollection_["strip_rh_rp1_st2_l2_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==3 and re==-1 and la==1) meCollection_["strip_rh_rm1_st3_l1_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==3 and re==-1 and la==2) meCollection_["strip_rh_rm1_st3_l2_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==3 and re== 1 and la==1) meCollection_["strip_rh_rp1_st3_l1_tot"]->Fill(gem_recHit_.firstClusterStrip);
	if(st==3 and re== 1 and la==2) meCollection_["strip_rh_rp1_st3_l2_tot"]->Fill(gem_recHit_.firstClusterStrip);
        
        /*--------------------------StripsVsRolls------------------------*/
	meCollection_["roll_vs_strip_rh"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
        
	if(st==1 and re==-1 and la==1) meCollection_["roll_vs_strip_rh_rm1_st1_l1"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==1 and re==-1 and la==2) meCollection_["roll_vs_strip_rh_rm1_st1_l2"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==1 and re== 1 and la==1) meCollection_["roll_vs_strip_rh_rp1_st1_l1"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==1 and re== 1 and la==2) meCollection_["roll_vs_strip_rh_rp1_st1_l2"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==2 and re==-1 and la==1) meCollection_["roll_vs_strip_rh_rm1_st2_l1"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==2 and re==-1 and la==2) meCollection_["roll_vs_strip_rh_rm1_st2_l2"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==2 and re== 1 and la==1) meCollection_["roll_vs_strip_rh_rp1_st2_l1"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==2 and re== 1 and la==2) meCollection_["roll_vs_strip_rh_rp1_st2_l2"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==3 and re==-1 and la==1) meCollection_["roll_vs_strip_rh_rm1_st3_l1"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==3 and re==-1 and la==2) meCollection_["roll_vs_strip_rh_rm1_st3_l2"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==3 and re== 1 and la==1) meCollection_["roll_vs_strip_rh_rp1_st3_l1"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
	if(st==3 and re== 1 and la==2) meCollection_["roll_vs_strip_rh_rp1_st3_l2"]->Fill(gem_recHit_.firstClusterStrip,gem_recHit_.roll);
        
	count++;
      }
    }
    gem_simHit_.countMatching = count;
  }
}
    

bool MuonGEMRecHits::isSimTrackGood(const SimTrack &t)
{
    // SimTrack selection
    if (t.noVertex()) return false;
    if (t.noGenpart()) return false;
    // only muons
    if (std::abs(t.type()) != 13) return false;
    // pt selection
    //if (t.momentum().pt() < simTrackMinPt_) return false;
    // eta selection
//     const float eta(std::abs(t.momentum().eta()));
//     if (eta > simTrackMaxEta_ || eta < simTrackMinEta_) return false;
    return true;
}

void MuonGEMRecHits::bookHistograms()
{
  int num_region=gem_geometry_->regions().size();
  int num_station=gem_geometry_->regions()[0]->stations().size();
  float nStrips=0;
  
  std::string region[2] ={"m1", "p1"};
  std::string station[3]={ "_st1", "_st2", "_st3" };
  
  meCollection_["clsDistribution"] = dbe_->book1D("clsDistribution","ClusterSizeDistribution",11,-0.5,10.5);
  
  meCollection_["bxDistribution"] = dbe_->book1D("bxDistribution","BunchCrossingDistribution",11,-5.5,5.5);
  meCollection_["recHitPullX"] = dbe_->book1D("recHitPullX","recHitPullX",100,-50,+50);
  meCollection_["recHitDPhi"] = dbe_->book1D("recHitDPhi","DeltaPhi RecHit",100,-0.001,+0.001);
  meCollection_["localrh_zr_rm1_st1"] = dbe_->book2D("localrh_zr_rm1_st1","GEM RecHit occupancy: region-1",200,-573,-564,110,130,240);
  meCollection_["localrh_zr_rp1_st1"] = dbe_->book2D("localrh_zr_rp1_st1","GEM RecHit occupancy: region1",200,564,573,110,130,240);
  meCollection_["strip_rh_tot"] = dbe_->book1D("strip_rh_tot","GEM RecHit occupancy per strip number",384,0.5,384.5);
  meCollection_["roll_vs_strip_rh"] = dbe_->book2D("roll_vs_strip_rh","GEM RecHit occupancy per roll and strip number",768,0.5,768.5,12,0.5,12.5);
  
  for (int k=0; k<num_station; k++){
    //-----------------------BunchX--------------------------------------//
    meCollection_["bxDistribution"+station[k]] = dbe_->book1D("bxDistribution"+station[k],"BunchCrossingDistribution, Station="+std::to_string(k+1),11,-5.5,5.5);
    //-----------------------Delta Phi--------------------------------------//
    meCollection_["recHitDPhi"+station[k]] = dbe_->book1D("recHitDPhi"+station[k],"DeltaPhi RecHit, Station="+std::to_string(k+1),100,-0.001,+0.001);
    meCollection_["recHitDPhi"+station[k]+"_cls1"] = dbe_->book1D("recHitDPhi"+station[k]+"_cls1","DeltaPhi RecHit, Station="+std::to_string(k+1)+", CLS=1",100,-0.001,+0.001);
    meCollection_["recHitDPhi"+station[k]+"_cls2"] = dbe_->book1D("recHitDPhi"+station[k]+"_cls2","DeltaPhi RecHit, Station="+std::to_string(k+1)+", CLS=2",100,-0.001,+0.001);
    meCollection_["recHitDPhi"+station[k]+"_cls3"] = dbe_->book1D("recHitDPhi"+station[k]+"_cls3","DeltaPhi RecHit, Station="+std::to_string(k+1)+", CLS=3",100,-0.001,+0.001);
  }
  
  for (int j=0; j<num_region; j++){
    
    meCollection_["clsDistribution_r"+region[j]+"_l1"] = dbe_->book1D("clsDistribution_r"+region[j]+"_l1","ClusterSizeDistribution, region "+region[j]+", Layer=1",11,-0.5,10.5);
    meCollection_["clsDistribution_r"+region[j]+"_l2"] = dbe_->book1D("clsDistribution_r"+region[j]+"_l2","ClusterSizeDistribution, region "+region[j]+", Layer=2",11,-0.5,10.5);
    
    for (int i=0; i<num_station; i++) {
      
      //-------------------------(x_rec-x_sim)/x_sim-----------------------------------//
      meCollection_["recHitPullX_r"+region[j]+station[i]+"_l1"] = dbe_->book1D("recHitPullX_r"+region[j]+station[i]+"_l1","recHitPullX, region "+region[j]+", station"+std::to_string(i+1)+", layer1",100,-50,+50);
      meCollection_["recHitPullX_r"+region[j]+station[i]+"_l2"] = dbe_->book1D("recHitPullX_r"+region[j]+station[i]+"_l2","recHitPullX, region "+region[j]+", station"+std::to_string(i+1)+", layer2",100,-50,+50);
      
      //----------------Occupancy XY-------------------------------//
      meCollection_["localrh_xy_r"+region[j]+station[i]+"_l1"] = dbe_->book2D("localrh_xy_r"+region[j]+station[i]+"_l1","GEM RecHit occupancy: region "+region[j]+", station"+std::to_string(i+1)+", layer1",200,-360,360,200,-360,360);
      meCollection_["localrh_xy_r"+region[j]+station[i]+"_l2"] = dbe_->book2D("localrh_xy_r"+region[j]+station[i]+"_l2","GEM RecHit occupancy: region"+region[j]+", station"+std::to_string(i+1)+", layer2",200,-360,360,200,-360,360);
      
      //---------------------Strips Occupancy------------------//
      
      if(i==0) nStrips=384.;  /*Station1*/
      if(i>0)  nStrips=768.;  /*Station2 & 3*/
      meCollection_["strip_rh_r"+region[j]+station[i]+"_l1_tot"] = dbe_->book1D("strip_rh_r"+region[j]+station[i]+"_l1_tot","GEM RecHit occupancy per strip number, region "+region[j]+" layer1 station"+std::to_string(i+1),nStrips,0.5,nStrips+0.5);
      meCollection_["strip_rh_r"+region[j]+station[i]+"_l2_tot"] = dbe_->book1D("strip_rh_r"+region[j]+station[i]+"_l2_tot","GEM RecHit occupancy per strip number, region "+region[j]+" layer2 station"+std::to_string(i+1),nStrips,0.5,nStrips+0.5);
      
      meCollection_["roll_vs_strip_rh_r"+region[j]+station[i]+"_l1"] = dbe_->book2D("roll_vs_strip_rh_r"+region[j]+station[i]+"_l1","GEM RecHit occupancy per roll and strip number, region "+region[j]+" layer1 station"+std::to_string(i+1),nStrips,0.5,nStrips+0.5,12,0.5,12.5);
      meCollection_["roll_vs_strip_rh_r"+region[j]+station[i]+"_l2"] = dbe_->book2D("roll_vs_strip_rh_r"+region[j]+station[i]+"_l2","GEM RecHit occupancy per roll and strip number, region "+region[j]+" layer2 station"+std::to_string(i+1),nStrips,0.5,nStrips+0.5,12,0.5,12.5);
      
    }
  }
}


void MuonGEMRecHits::beginJob()
{
}


void MuonGEMRecHits::endJob()
{
}


void 
MuonGEMRecHits::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  try {
    iSetup.get<MuonGeometryRecord>().get(gem_geom_);
    gem_geometry_ = &*gem_geom_;
    hasGEMGeometry_ = true;
  } catch (edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    hasGEMGeometry_ = false;
    edm::LogWarning("MuonGEMRecHits") << "+++ Info: GEM geometry is unavailable. +++\n";
  }

  dbe_->setCurrentFolder(folderPath_);

  if(hasGEMGeometry_) {
    bookHistograms();
  }
}


void 
MuonGEMRecHits::endRun(edm::Run const&, edm::EventSetup const&)
{
  if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}


void
MuonGEMRecHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGEMRecHits);
