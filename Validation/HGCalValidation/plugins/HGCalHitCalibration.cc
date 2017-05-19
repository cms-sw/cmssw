// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "TH1F.h"
#include <string>
#include <map>

//#define EDM_ML_DEBUG

class HGCalHitCalibration : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit HGCalHitCalibration(const edm::ParameterSet&);
  ~HGCalHitCalibration();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override {}

  edm::EDGetTokenT<HGCRecHitCollection> _recHitsEE;
  edm::EDGetTokenT<HGCRecHitCollection> _recHitsFH;
  edm::EDGetTokenT<HGCRecHitCollection> _recHitsBH;
  edm::EDGetTokenT<std::vector<CaloParticle> > _caloParticles;

  bool                       rawRecHits_, cutOnPt_;
  double                     cutValue_;
  int                        algo_;
  hgcal::RecHitTools         recHitTools;

  TH1F* h_EoP_CPene_100_calib_fraction_[6];
  TH1F* h_EoP_CPene_200_calib_fraction_[6];
  TH1F* h_EoP_CPene_300_calib_fraction_[6];
  TH1F* h_LayerOccupancy_[6];
  TH1F* h_MissingHit_[3];

  std::vector<float> Energy_layer_calib;
  std::vector<float> Energy_layer_calib_fraction;
};

HGCalHitCalibration::HGCalHitCalibration(const edm::ParameterSet& iConfig) :
  rawRecHits_(iConfig.getParameter<bool>("rawRecHits")),
  cutOnPt_(iConfig.getParameter<bool>("cutOnPt")),
  cutValue_(iConfig.getParameter<double>("cutValue")) {

  usesResource(TFileService::kSharedResource);
  std::string detector = iConfig.getParameter<std::string >("detector");
  _recHitsEE = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCEERecHits"));
  _recHitsFH = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCHEFRecHits"));
  _recHitsBH = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCHEBRecHits"));
  if (detector=="all") {
    algo_ = 1;
  } else if (detector=="EM") {
    algo_ = 2;
  } else if (detector=="HAD") {
    algo_ = 3;
  }
  _caloParticles = consumes<std::vector<CaloParticle> >(edm::InputTag("mix","MergedCaloTruth"));

  edm::Service<TFileService> fs;
  std::string particle[6] = {"All", "Electron", "Muon", "Photon", "ChgHad", "NeutHad"};
  char name[100];
  for (int k=0; k<6; ++k) {
    sprintf(name,"h_EoP_CPene_100_calib_fraction_%s",particle[k].c_str());
    h_EoP_CPene_100_calib_fraction_[k] = fs->make<TH1F>(name,"",1000,-0.5,2.5);
    sprintf(name,"h_EoP_CPene_200_calib_fraction_%s",particle[k].c_str());
    h_EoP_CPene_200_calib_fraction_[k] = fs->make<TH1F>(name,"",1000,-0.5,2.5);
    sprintf(name,"h_EoP_CPene_300_calib_fraction_%s",particle[k].c_str());
    h_EoP_CPene_300_calib_fraction_[k] = fs->make<TH1F>(name,"",1000,-0.5,2.5);
    sprintf(name,"h_LayerOccupancy_%s",particle[k].c_str());
    h_LayerOccupancy_[k] = fs->make<TH1F>(name, "", 60, 0., 60.);
  }
  std::string dets[3] = {"EE", "FH", "BH"};
  for (int k=0; k<3; ++k) {
    sprintf(name,"h_missHit_%s",dets[k].c_str());
    h_MissingHit_[k] = fs->make<TH1F>(name, "", 200, 0., 200.);
  }
}

HGCalHitCalibration::~HGCalHitCalibration() { }

void HGCalHitCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  recHitTools.getEventSetup(iSetup);

  edm::Handle<HGCRecHitCollection> recHitHandleEE;
  edm::Handle<HGCRecHitCollection> recHitHandleFH;
  edm::Handle<HGCRecHitCollection> recHitHandleBH;
  iEvent.getByToken(_recHitsEE,recHitHandleEE);
  iEvent.getByToken(_recHitsFH,recHitHandleFH);
  iEvent.getByToken(_recHitsBH,recHitHandleBH);
  const HGCRecHitCollection& rechitsEE = *recHitHandleEE;
  const HGCRecHitCollection& rechitsFH = *recHitHandleFH;
  const HGCRecHitCollection& rechitsBH = *recHitHandleBH;
  
  edm::Handle<std::vector<CaloParticle> > caloParticleHandle;
  iEvent.getByToken(_caloParticles, caloParticleHandle);
  const std::vector<CaloParticle>& caloParticles = *caloParticleHandle;
  const int pdgId[3] = {11, 13, 22};

  // loop over caloParticles
  int mhit[3] = {0,0,0};
  for (auto it_caloPart = caloParticles.begin();
       it_caloPart != caloParticles.end(); ++it_caloPart) {
    double cut = (cutOnPt_) ? it_caloPart->pt() : it_caloPart->energy();
    if (cut > cutValue_) {
      int type(5);
      if      (std::abs(it_caloPart->pdgId()) == pdgId[0]) type = 1;
      else if (std::abs(it_caloPart->pdgId()) == pdgId[1]) type = 2;
      else if (it_caloPart->pdgId()           == pdgId[2]) type = 3;
      else if (it_caloPart->threeCharge()     != 0)        type = 4;
      const SimClusterRefVector simClusterRefVector = it_caloPart->simClusters();

      //size should be HGC layers 52 is enough
      Energy_layer_calib.assign(60,0.);
      Energy_layer_calib_fraction.assign(60,0.);
    
      int seedDet = 0;
      float seedEnergy = 0.;
      int simClusterCount = 0;
    
      for ( const auto & simCluster : simClusterRefVector) {
	++simClusterCount;
#ifdef EDM_ML_DEBUG
	std::cout << ">>> simCluster.energy() = " << simCluster->energy() << std::endl;
#endif
	const std::vector<std::pair<uint32_t,float> > hits_and_fractions = simCluster->hits_and_fractions();

	//loop over hits      
	for (auto it_haf = hits_and_fractions.begin(); 
	     it_haf != hits_and_fractions.end(); ++it_haf) {
	  unsigned int hitlayer = recHitTools.getLayerWithOffset(it_haf->first);
	  DetId hitid = (it_haf->first); 

	  // dump raw RecHits and match
	  if (rawRecHits_) {
	    if (hitid.det() == DetId::Forward && hitid.subdetId() == HGCEE &&
		(algo_ == 1 || algo_ == 2)) {
	      // loop over EE RecHits
	      bool found(false);
	      for (auto it_hit = rechitsEE.begin(); 
		   it_hit < rechitsEE.end(); ++it_hit) {
		const HGCalDetId detid = it_hit->detid();
		unsigned int layer = recHitTools.getLayerWithOffset(detid);
		if (detid == hitid) {
		  found = true;
		  if (hitlayer != layer) {
#ifdef EDM_ML_DEBUG
		    std::cout << " recHit ID problem EE " << std::endl;
#endif
		    return;
		  }
		  Energy_layer_calib_fraction[layer] += it_hit->energy()*it_haf->second;
		  h_LayerOccupancy_[0]->Fill(layer);
		  h_LayerOccupancy_[type]->Fill(layer);
		  if(seedEnergy < it_hit->energy()){
		    seedEnergy = it_hit->energy();
		    seedDet = recHitTools.getSiThickness(detid);
		  }
		  break;
		}
	      }
	      if (!found) ++mhit[0];
	    }
	    if (hitid.det() == DetId::Forward && hitid.subdetId() == HGCHEF &&
		(algo_ == 1 || algo_ == 3)) {
	      // loop over HEF RecHits
	      bool found(false);
	      for (auto it_hit = rechitsFH.begin(); 
		   it_hit < rechitsFH.end(); ++it_hit) {
		const HGCalDetId detid = it_hit->detid();
		unsigned int layer = recHitTools.getLayerWithOffset(detid);
		if (detid == hitid) {
		  found = true;
		  if (hitlayer != layer) {
#ifdef EDM_ML_DEBUG
		    std::cout << " recHit ID problem FH " << std::endl; 
#endif
		    return;
		  }
		  Energy_layer_calib_fraction[layer] += it_hit->energy()*it_haf->second;
		  h_LayerOccupancy_[0]->Fill(layer);
		  h_LayerOccupancy_[type]->Fill(layer);
		  if(seedEnergy < it_hit->energy()){
		    seedEnergy = it_hit->energy();
		    seedDet = recHitTools.getSiThickness(detid);
		  }
		  break;
		}
	      }
	      if (!found) ++mhit[1];
	    }
	    if (hitid.det() == DetId::Hcal && hitid.subdetId() == HcalEndcap &&
		(algo_ == 1 || algo_ == 3)) {
	      // loop over BH RecHits
	      bool found(false);
	      for (auto it_hit = rechitsBH.begin();
		   it_hit < rechitsBH.end(); ++it_hit) {
		const HcalDetId detid = it_hit->detid();
		unsigned int layer = recHitTools.getLayerWithOffset(detid);
		if (detid == hitid) {
		  found = true;
		  if (hitlayer != layer) {
#ifdef EDM_ML_DEBUG
		    std::cout << " recHit ID problem BH " << std::endl; 
#endif
		    return;
		  }
		  Energy_layer_calib_fraction[layer] += it_hit->energy()*it_haf->second;
		  h_LayerOccupancy_[0]->Fill(layer);
		  h_LayerOccupancy_[type]->Fill(layer);
		  if (seedEnergy < it_hit->energy()){
		    seedEnergy = it_hit->energy();
		    seedDet = 300; // recHitTools.getSiThickness(detid);
		  }
		  break;
		}
	      }
	      if (!found) ++mhit[2];
	    }
	  
	  }//end recHits
	}// end simHit
      }//end simCluster

      float sumCalibRecHitCalib_fraction = 0;
      for(unsigned int iL=0; iL<Energy_layer_calib_fraction.size(); ++iL){
	sumCalibRecHitCalib_fraction += Energy_layer_calib_fraction[iL];
      }
    
      double ebyp = sumCalibRecHitCalib_fraction / it_caloPart->energy();
      if(seedDet == 100) {
	h_EoP_CPene_100_calib_fraction_[0]->Fill(ebyp);
	h_EoP_CPene_100_calib_fraction_[type]->Fill(ebyp);
      } else if (seedDet == 200) {
	h_EoP_CPene_200_calib_fraction_[0]->Fill(ebyp);
	h_EoP_CPene_200_calib_fraction_[type]->Fill(ebyp);
      } else if (seedDet == 300){
	h_EoP_CPene_300_calib_fraction_[0]->Fill(ebyp);
	h_EoP_CPene_300_calib_fraction_[type]->Fill(ebyp);
      }
    }
  }//end caloparticle

  for (int k=0; k<3; ++k) h_MissingHit_[k]->Fill(mhit[k]);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HGCalHitCalibration::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector","all");
  desc.add<bool>("rawRecHits",true);
  desc.add<bool>("cutOnPt",true);
  desc.add<double>("cutValue",10.0);
  descriptions.add("hgcalHitCalibration",desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalHitCalibration);
