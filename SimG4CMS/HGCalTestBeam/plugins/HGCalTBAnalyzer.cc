// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

//#define DebugLog

class HGCalTBAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit HGCalTBAnalyzer(edm::ParameterSet const&);
  ~HGCalTBAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  void analyzeSimHits (int type, std::vector<PCaloHit>& hits);
  template<class T1> void analyzeDigi (int type, const T1& detId, uint16_t adc);
  void analyzeRecHits (int type, edm::Handle<HGCRecHitCollection> & hits);

  edm::Service<TFileService>               fs_;
  const HGCalDDDConstants                 *hgcons_[2];
  const HGCalGeometry                     *hgeom_[2];
  bool                                     ifEE_, ifHE_;
  bool                                     doSimHits_, doDigis_, doRecHits_;
  std::string                              detectorEE_, detectorHE_;
  int                                      sampleIndex_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hitsEE_, tok_hitsHE_;
  edm::EDGetToken                          tok_digiEE_, tok_digiHE_;
  edm::EDGetToken                          tok_hitrEE_, tok_hitrHE_;
  edm::EDGetTokenT<edm::HepMCProduct>      tok_hepMC_;
  TH1D                                    *hSimHitE_[2], *hSimHitT_[2];
  TH1D                                    *hSimHitLng_[2], *hBeam_;
  TH1D                                    *hDigiADC_[2], *hDigiLng_[2];
  TH1D                                    *hRecHitE_[2], *hRecHitLng_[2];
  TH2D                                    *hSimHitLat_[2], *hRecHitOcc_[2];
  TH2D                                    *hDigiOcc_[2];
};

HGCalTBAnalyzer::HGCalTBAnalyzer(const edm::ParameterSet& iConfig) {

  usesResource("TFileService");

  //now do whatever initialization is needed
  detectorEE_  = iConfig.getParameter<std::string>("DetectorEE");
  detectorHE_  = iConfig.getParameter<std::string>("DetectorHE");
  ifEE_        = iConfig.getParameter<bool>("UseEE");
  ifHE_        = iConfig.getParameter<bool>("UseHE");
  doSimHits_   = iConfig.getParameter<bool>("DoSimHits");
  doDigis_     = iConfig.getParameter<bool>("DoDigis");
  sampleIndex_ = iConfig.getParameter<int>("SampleIndex");
  doRecHits_   = iConfig.getParameter<bool>("DoRecHits");
#ifdef DebugLog
  std::cout << "HGCalTBAnalyzer:: SimHits = " << doSimHits_ << " Digis = "
	    << doDigis_ << ":" << sampleIndex_ << " RecHits = " << doRecHits_
	    << " useDets " << ifEE_ << ":" << ifHE_ << std::endl;
#endif

  edm::InputTag tmp0 = iConfig.getParameter<edm::InputTag>("GeneratorSrc");
  tok_hepMC_   = consumes<edm::HepMCProduct>(tmp0);
#ifdef DebugLog
  std::cout << "HGCalTBAnalyzer:: GeneratorSource = " << tmp0 << std::endl;
#endif
  std::string   tmp1 = iConfig.getParameter<std::string>("CaloHitSrcEE");
  tok_hitsEE_  = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",tmp1));
  edm::InputTag tmp2 = iConfig.getParameter<edm::InputTag>("DigiSrcEE");
  tok_digiEE_  = consumes<HGCEEDigiCollection>(tmp2);
  edm::InputTag tmp3 = iConfig.getParameter<edm::InputTag>("RecHitSrcEE");
  tok_hitrEE_  = consumes<HGCRecHitCollection>(tmp3);
#ifdef DebugLog
  if (ifEE_) {
    std::cout << "HGCalTBAnalyzer:: Detector " << detectorEE_ << " with tags "
	      << tmp1 << ", " << tmp2 << ", " << tmp3 << std::endl;
  }
#endif
  tmp1         = iConfig.getParameter<std::string>("CaloHitSrcHE");
  tok_hitsHE_  = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",tmp1));
  tmp2         = iConfig.getParameter<edm::InputTag>("DigiSrcHE");
  tok_digiHE_  = consumes<HGCHEDigiCollection>(tmp2);
  tmp3         = iConfig.getParameter<edm::InputTag>("RecHitSrcHE");
  tok_hitrHE_  = consumes<HGCRecHitCollection>(tmp3);
#ifdef DebugLog
  if (ifHE_) {
    std::cout << "HGCalTBAnalyzer:: Detector " << detectorHE_ << " with tags "
	      << tmp1 << ", " << tmp2 << ", " << tmp3 << std::endl;
  }
#endif
}

HGCalTBAnalyzer::~HGCalTBAnalyzer() {}

void HGCalTBAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HGCalTBAnalyzer::beginJob() {

  char name[40], title[100];
  hBeam_ = fs_->make<TH1D>("BeamP", "Beam Momentum", 1000, 0, 1000.0);
  for (int i=0; i<2; ++i) {
    bool book = (i == 0) ? ifEE_ : ifHE_;
    std::string det = (i == 0) ? detectorEE_ : detectorHE_;

    if (doSimHits_ && book) {
      sprintf (name, "SimHitEn%s", det.c_str());
      sprintf (title,"Sim Hit Energy for %s", det.c_str());
      hSimHitE_[i] = fs_->make<TH1D>(name,title,5000,0.,10.0);
      sprintf (name, "SimHitTm%s", det.c_str());
      sprintf (title,"Sim Hit Timing for %s", det.c_str());
      hSimHitT_[i] = fs_->make<TH1D>(name,title,5000,0.,500.0);
      sprintf (name, "SimHitLat%s", det.c_str());
      sprintf (title,"Lateral Shower profile (Sim Hit)for %s", det.c_str());
      hSimHitLat_[i] = fs_->make<TH2D>(name,title,100,-100.,100.,100,-100.,100.);
      sprintf (name, "SimHitLng%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile (Sim Hit)for %s",det.c_str());
      hSimHitLng_[i] = fs_->make<TH1D>(name,title,50,0.,100.);
    }

    if (doDigis_ && book) {
      sprintf (name, "DigiADC%s", det.c_str());
      sprintf (title,"ADC at Digi level for %s", det.c_str());
      hDigiADC_[i] = fs_->make<TH1D>(name,title,100,0.,100.0);
      sprintf (name, "DigiOcc%s", det.c_str());
      sprintf (title,"Occupancy (Digi)for %s", det.c_str());
      hDigiOcc_[i] = fs_->make<TH2D>(name,title,100,-10.,10.,100,-10.,10.);
      sprintf (name, "DigiLng%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile (Digi)for %s",det.c_str());
      hDigiLng_[i] = fs_->make<TH1D>(name,title,100,0.,10.);
    }

    if (doRecHits_ && book) {
      sprintf (name, "RecHitEn%s", det.c_str());
      sprintf (title,"Rec Hit Energy for %s", det.c_str());
      hRecHitE_[i] = fs_->make<TH1D>(name,title,1000,0.,100.0);
      sprintf (name, "RecHitOcc%s", det.c_str());
      sprintf (title,"Occupancy (Rec Hit)for %s", det.c_str());
      hRecHitOcc_[i] = fs_->make<TH2D>(name,title,100,-10.,10.,100,-10.,10.);
      sprintf (name, "RecHitLng%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile (Rec Hit)for %s",det.c_str());
      hRecHitLng_[i] = fs_->make<TH1D>(name,title,100,0.,10.);
    }
  }
}

void HGCalTBAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {

  if (ifEE_) {
    edm::ESHandle<HGCalDDDConstants>  pHGDC;
    iSetup.get<IdealGeometryRecord>().get(detectorEE_, pHGDC);
    hgcons_[0] = &(*pHGDC);
    if (doDigis_ || doRecHits_) {
      edm::ESHandle<HGCalGeometry> geom;
      iSetup.get<IdealGeometryRecord>().get(detectorEE_, geom);
      hgeom_[0] = geom.product();
    } else {
      hgeom_[0] = 0;
    }
#ifdef DebugLog
    std::cout << "HGCalTBAnalyzer::" << detectorEE_ << " defined with "
	      << hgcons_[0]->layers(false) << " layers" << std::endl;
#endif
  } else {
    hgcons_[0] = 0;
    hgeom_[0]  = 0;
  }

  if (ifHE_) {
    edm::ESHandle<HGCalDDDConstants>  pHGDC;
    iSetup.get<IdealGeometryRecord>().get(detectorHE_, pHGDC);
    hgcons_[1] = &(*pHGDC);
    if (doDigis_ || doRecHits_) {
      edm::ESHandle<HGCalGeometry> geom;
      iSetup.get<IdealGeometryRecord>().get(detectorHE_, geom);
      hgeom_[1] = geom.product();
    } else {
      hgeom_[1] = 0;
    }
#ifdef DebugLog
    std::cout << "HGCalTBAnalyzer::" << detectorHE_ << " defined with "
	      << hgcons_[1]->layers(false) << " layers" << std::endl;
#endif
  } else {
    hgcons_[1] = 0;
    hgeom_[1]  = 0;
  }
}

void HGCalTBAnalyzer::analyze(const edm::Event& iEvent, 
			      const edm::EventSetup& iSetup) {

  //Generator input
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_hepMC_,evtMC);
  if (!evtMC.isValid()) {
    edm::LogWarning("HGCal") << "no HepMCProduct found";
  } else { 
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    unsigned int k(0);
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
         p != myGenEvent->particles_end(); ++p, ++k) {
      if (k == 0) hBeam_->Fill((*p)->momentum().rho());
#ifdef DebugLog
      std::cout << "Particle[" << k << "] with p " << (*p)->momentum().rho() 
		<< " theta " << (*p)->momentum().theta() << " phi "
		<< (*p)->momentum().phi() << std::endl;
#endif
    }
  }

  //Now the Simhits
  if (doSimHits_) {
    edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
    std::vector<PCaloHit>               caloHits;
    if (ifEE_) {
      iEvent.getByToken(tok_hitsEE_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef DebugLog
	std::cout << "PcalohitContainer for " << detectorEE_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	caloHits.clear();
	caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
			theCaloHitContainers->end());
	analyzeSimHits(0, caloHits);
      } else {
#ifdef DebugLog
	std::cout << "PCaloHitContainer does not exist for " << detectorEE_ 
		  << " !!!" << std::endl;
#endif
      }
    }
    if (ifHE_) {
      iEvent.getByToken(tok_hitsHE_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef DebugLog
	std::cout << "PcalohitContainer for " << detectorHE_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	caloHits.clear();
	caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
			theCaloHitContainers->end());
	analyzeSimHits(1, caloHits);
      } else {
#ifdef DebugLog
	std::cout << "PCaloHitContainer does not exist for " << detectorHE_ 
		  << " !!!" << std::endl;
#endif
      }
    }
  }

  //Now the Digis
  if (doDigis_) {
    if (ifEE_) {
      edm::Handle<HGCEEDigiCollection> theDigiContainers;
      iEvent.getByToken(tok_digiEE_, theDigiContainers);
      if (theDigiContainers.isValid()) {
#ifdef DebugLog
	std::cout << "HGCDigiCintainer for " << detectorEE_ << " with " 
		  << theDigiContainers->size() << " element(s)" << std::endl;
#endif
	for (HGCEEDigiCollection::const_iterator it =theDigiContainers->begin();
	     it !=theDigiContainers->end(); ++it) {
	  HGCEEDetId detId     = (it->id());
	  HGCSample  hgcSample = it->sample(sampleIndex_);
	  uint16_t   adc       = hgcSample.data();
	  analyzeDigi(0, detId, adc);
	}
      }
    }
    if (ifHE_) {
      edm::Handle<HGCHEDigiCollection> theDigiContainers;
      iEvent.getByToken(tok_digiHE_, theDigiContainers);
      if (theDigiContainers.isValid()) {
#ifdef DebugLog
	std::cout << "HGCDigiContainer for " << detectorHE_ << " with " 
		  << theDigiContainers->size() << " element(s)" << std::endl;
#endif
	for (HGCHEDigiCollection::const_iterator it =theDigiContainers->begin();
	     it !=theDigiContainers->end(); ++it) {
	  HGCHEDetId detId     = (it->id());
	  HGCSample  hgcSample = it->sample(sampleIndex_);
	  uint16_t   adc       = hgcSample.data();
	  analyzeDigi(1, detId, adc);
	}
      }
    }
  }

  //The Rechits
  if (doRecHits_) {
    edm::Handle<HGCRecHitCollection> theCaloHitContainers;
    if (ifEE_) {
      iEvent.getByToken(tok_hitrEE_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef DebugLog
	std::cout << "HGCRecHitCollection for " << detectorEE_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	analyzeRecHits(0, theCaloHitContainers);
      } else {
#ifdef DebugLog
	std::cout << "HGCRecHitCollection does not exist for " << detectorEE_ 
		  << " !!!" << std::endl;
#endif
      }
    }
    if (ifHE_) {
      iEvent.getByToken(tok_hitrHE_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef DebugLog
	std::cout << "HGCRecHitCollection for " << detectorHE_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	analyzeRecHits(1, theCaloHitContainers);
      } else {
#ifdef DebugLog
	std::cout << "HGCRecHitCollection does not exist for " << detectorHE_ 
		  << " !!!" << std::endl;
#endif
      }
    }
  }
}

void HGCalTBAnalyzer::analyzeSimHits (int type, std::vector<PCaloHit>& hits) {

  std::map<uint32_t,double> map_hits;
  map_hits.clear();
  for (unsigned int i=0; i<hits.size(); i++) {
    double energy      = hits[i].energy();
    double time        = hits[i].time();
    uint32_t id        = hits[i].id();
    if (map_hits.count(id) != 0) {
      map_hits[id] += energy;
    } else {
      map_hits[id]  = energy;
    }
    hSimHitE_[type]->Fill(energy);
    hSimHitT_[type]->Fill(time,energy);
  }

  for (std::map<uint32_t,double>::iterator itr = map_hits.begin() ; 
       itr != map_hits.end(); ++itr)   {
    uint32_t id       = itr->first;
    double   energy   = itr->second;
    int      subdet, zside, layer, sector, subsector, cell;
    HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, sector,
					   subsector, cell);
    std::pair<float,float> xy = hgcons_[type]->locateCell(cell,layer,sector,false);
    double zp = hgcons_[type]->waferZ(layer,false);
    double xx = (zp < 0) ? -xy.first : xy.first;
    HepGeom::Point3D<float> gcoord  = HepGeom::Point3D<float>(xx,xy.second,zp);
    hSimHitLat_[type]->Fill(xx,xy.second,energy);
    hSimHitLng_[type]->Fill(zp,energy);
  }
}

template<class T1>
void HGCalTBAnalyzer::analyzeDigi (int type, const T1& detId, uint16_t adc) {

  DetId id1 = DetId(detId.rawId());
  GlobalPoint global = hgeom_[type]->getPosition(id1);
  hDigiOcc_[type]->Fill(global.x(),global.y());
  hDigiLng_[type]->Fill(global.z());
  hDigiADC_[type]->Fill(adc);
}

void HGCalTBAnalyzer::analyzeRecHits (int type, 
				      edm::Handle<HGCRecHitCollection>& hits) {
 
  for (HGCRecHitCollection::const_iterator it = hits->begin(); 
       it != hits->end(); ++it) {
    DetId       detId  = it->id();
    GlobalPoint global = hgeom_[type]->getPosition(detId);
    double      energy = it->energy();
    hRecHitOcc_[type]->Fill(global.x(),global.y());
    hRecHitLng_[type]->Fill(global.z());
    hRecHitE_[type]->Fill(energy);
  }
}
  
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTBAnalyzer);
