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

  std::string                detector;
  int                        algo;
  HGCalDepthPreClusterer     pre;
  bool                       rawRecHits;
  hgcal::RecHitTools         recHitTools;

  TH1F* h_EoP_CPene_100_calib_fraction;
  TH1F* h_EoP_CPene_200_calib_fraction;
  TH1F* h_EoP_CPene_300_calib_fraction;
  TH1F* LayerOccupancy;

  std::vector<float> Energy_layer_calib;
  std::vector<float> Energy_layer_calib_fraction;
};



HGCalHitCalibration::HGCalHitCalibration(const edm::ParameterSet& iConfig) :
  detector(iConfig.getParameter<std::string >("detector")),
  rawRecHits(iConfig.getParameter<bool>("rawRecHits")) {

  usesResource(TFileService::kSharedResource);

  if(detector=="all") {
    _recHitsEE = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCEERecHits"));
    _recHitsFH = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCHEFRecHits"));
    _recHitsBH = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCHEBRecHits"));
    algo = 1;
  }else if(detector=="EM") {
    _recHitsEE = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCEERecHits"));
    algo = 2;
  }else if(detector=="HAD") {
    _recHitsFH = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCHEFRecHits"));
    _recHitsBH = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit","HGCHEBRecHits"));
    algo = 3;
  }
  _caloParticles = consumes<std::vector<CaloParticle> >(edm::InputTag("mix","MergedCaloTruth"));

  edm::Service<TFileService> fs;
  h_EoP_CPene_100_calib_fraction = fs->make<TH1F>("h_EoP_CPene_100_calib_fraction", "", 1000, -0.5, 2.5);
  h_EoP_CPene_200_calib_fraction = fs->make<TH1F>("h_EoP_CPene_200_calib_fraction", "", 1000, -0.5, 2.5);
  h_EoP_CPene_300_calib_fraction = fs->make<TH1F>("h_EoP_CPene_300_calib_fraction", "", 1000, -0.5, 2.5);
  LayerOccupancy = fs->make<TH1F>("LayerOccupancy", "", 60, 0., 60.);
}

HGCalHitCalibration::~HGCalHitCalibration() {

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


void
HGCalHitCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //size should be HGC layers 52 is enough
  Energy_layer_calib.clear();
  Energy_layer_calib_fraction.clear();
  for(unsigned int ij=0; ij<60; ++ij){
    Energy_layer_calib.push_back(0.);
    Energy_layer_calib_fraction.push_back(0.);
  }

  recHitTools.getEventSetup(iSetup);

  edm::Handle<HGCRecHitCollection> recHitHandleEE;
  edm::Handle<HGCRecHitCollection> recHitHandleFH;
  edm::Handle<HGCRecHitCollection> recHitHandleBH;
  
  edm::Handle<std::vector<CaloParticle> > caloParticleHandle;
  iEvent.getByToken(_caloParticles, caloParticleHandle);
  const std::vector<CaloParticle>& caloParticles = *caloParticleHandle;
  
  //make a map detid-rechit
  std::map<DetId,const HGCRecHit*> hitmap;
  switch(algo){
  case 1:
    {
      iEvent.getByToken(_recHitsEE,recHitHandleEE);
      iEvent.getByToken(_recHitsFH,recHitHandleFH);
      iEvent.getByToken(_recHitsBH,recHitHandleBH);
      const auto& rechitsEE = *recHitHandleEE;
      const auto& rechitsFH = *recHitHandleFH;
      const auto& rechitsBH = *recHitHandleBH;
      for(unsigned int i = 0; i < rechitsEE.size(); ++i){
	hitmap[rechitsEE[i].detid()] = &rechitsEE[i];
      }
      for(unsigned int i = 0; i < rechitsFH.size(); ++i){
	hitmap[rechitsFH[i].detid()] = &rechitsFH[i];
      }
      for(unsigned int i = 0; i < rechitsBH.size(); ++i){
	hitmap[rechitsBH[i].detid()] = &rechitsBH[i];
      }
      break;
    }
  case 2:
    {
      iEvent.getByToken(_recHitsEE,recHitHandleEE);
      const HGCRecHitCollection& rechitsEE = *recHitHandleEE;
      for(unsigned int i = 0; i < rechitsEE.size(); i++){
	hitmap[rechitsEE[i].detid()] = &rechitsEE[i];
      }
      break;
    }
  case 3:
    {
      iEvent.getByToken(_recHitsFH,recHitHandleFH);
      iEvent.getByToken(_recHitsBH,recHitHandleBH);
      const auto& rechitsFH = *recHitHandleFH;
      const auto& rechitsBH = *recHitHandleBH;
      for(unsigned int i = 0; i < rechitsFH.size(); i++){
	hitmap[rechitsFH[i].detid()] = &rechitsFH[i];
      }
      for(unsigned int i = 0; i < rechitsBH.size(); i++){
	hitmap[rechitsBH[i].detid()] = &rechitsBH[i];
      }
      break;
    }
  default:
    break;
  }

  // loop over caloParticles
  for (std::vector<CaloParticle>::const_iterator it_caloPart = caloParticles.begin(); it_caloPart != caloParticles.end(); ++it_caloPart) {
    const SimClusterRefVector simClusterRefVector = it_caloPart->simClusters();

    Energy_layer_calib.clear();
    Energy_layer_calib_fraction.clear();
    for(unsigned int ij=0; ij<60; ++ij) {
      Energy_layer_calib.push_back(0.);
      Energy_layer_calib_fraction.push_back(0.);
    }

    int seedDet = 0;
    float seedEnergy = 0.;
    int simClusterCount = 0;
    
    for (CaloParticle::sc_iterator it_sc = simClusterRefVector.begin(); it_sc != simClusterRefVector.end(); ++it_sc) {
      const SimCluster simCluster = (*(*it_sc));
      ++simClusterCount;
      //      std::cout << ">>> simCluster.energy() = " << simCluster.energy() << std::endl;
      const std::vector<std::pair<uint32_t,float> > hits_and_fractions = simCluster.hits_and_fractions();

      //loop over hits      
      for (std::vector<std::pair<uint32_t,float> >::const_iterator it_haf = hits_and_fractions.begin(); it_haf != hits_and_fractions.end(); ++it_haf) {
	unsigned int hitlayer = recHitTools.getLayerWithOffset(it_haf->first);
	DetId hitid = (it_haf->first); 
	
        // dump raw RecHits and match
	if (rawRecHits) {
	  if (hitid.det() == DetId::Forward && hitid.subdetId() == HGCEE) {
	    const HGCRecHitCollection& rechitsEE = *recHitHandleEE;
	    // loop over EE RecHits
	    for (HGCRecHitCollection::const_iterator it_hit = rechitsEE.begin(); it_hit < rechitsEE.end(); ++it_hit) {
	      const HGCalDetId detid = it_hit->detid();
	      unsigned int layer = recHitTools.getLayerWithOffset(detid);
	      if(detid == hitid){
		if(hitlayer != layer) {std::cout << " recHit ID problem EE " << std::endl; return;}
		Energy_layer_calib_fraction[layer] += it_hit->energy()*it_haf->second;
		LayerOccupancy->Fill(layer);
		if(seedEnergy < it_hit->energy()){
		  seedEnergy = it_hit->energy();
		  seedDet = recHitTools.getSiThickness(detid);
		}
		break;
	      }
	    }
	  }
	  if (hitid.det() == DetId::Forward && hitid.subdetId() == HGCHEF) {
	    const HGCRecHitCollection& rechitsFH = *recHitHandleFH;
	    // loop over HEF RecHits
	    for (HGCRecHitCollection::const_iterator it_hit = rechitsFH.begin(); it_hit < rechitsFH.end(); ++it_hit) {
	      const HGCalDetId detid = it_hit->detid();
	      unsigned int layer = recHitTools.getLayerWithOffset(detid);
	      if(detid == hitid){
		if(hitlayer != layer) {std::cout << " recHit ID problem FH " << std::endl; return;}
		Energy_layer_calib_fraction[layer] += it_hit->energy()*it_haf->second;
		LayerOccupancy->Fill(layer);
		if(seedEnergy < it_hit->energy()){
		  seedEnergy = it_hit->energy();
		  seedDet = recHitTools.getSiThickness(detid);
		}
		break;
	      }
	    }
	  }
	  if (hitid.det() == DetId::Forward && hitid.subdetId() == HGCHEB) {
	    const HGCRecHitCollection& rechitsBH = *recHitHandleBH;
	    // loop over BH RecHits
	    for (HGCRecHitCollection::const_iterator it_hit = rechitsBH.begin(); it_hit < rechitsBH.end(); ++it_hit) {
	      const HcalDetId detid = it_hit->detid();
	      unsigned int layer = recHitTools.getLayerWithOffset(detid);
	      if(detid == hitid){
		if(hitlayer != layer) {std::cout << " recHit ID problem BH " << std::endl; return;}
		Energy_layer_calib_fraction[layer] += it_hit->energy()*it_haf->second;
		LayerOccupancy->Fill(layer);
		if(seedEnergy < it_hit->energy()){
		  seedEnergy = it_hit->energy();
		  seedDet = recHitTools.getSiThickness(detid);
		}
		break;
	      }
	    }
	  }
	  
	}//end recHits
      }// end simHit
    }//end simCluster


    float sumCalibRecHitCalib_fraction = 0;
    for(unsigned int iL=0; iL<Energy_layer_calib_fraction.size(); ++iL){
      sumCalibRecHitCalib_fraction += Energy_layer_calib_fraction[iL];
    }
    
    if(seedDet == 100){
      h_EoP_CPene_100_calib_fraction->Fill(sumCalibRecHitCalib_fraction / it_caloPart->energy());
    }
    if(seedDet == 200){
      h_EoP_CPene_200_calib_fraction->Fill(sumCalibRecHitCalib_fraction / it_caloPart->energy());
    }
    if(seedDet == 300){
      h_EoP_CPene_300_calib_fraction->Fill(sumCalibRecHitCalib_fraction / it_caloPart->energy());
    }

  }//end caloparticle


}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HGCalHitCalibration::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector","all");
  desc.add<bool>("rawRecHits",true);
  descriptions.add("hgcalHitCalibration",desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalHitCalibration);
