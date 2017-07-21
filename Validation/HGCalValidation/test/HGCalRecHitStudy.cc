// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "TH1D.h"
#include "TH2D.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class HGCalRecHitStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:

  explicit HGCalRecHitStudy(const edm::ParameterSet&);
  ~HGCalRecHitStudy();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  struct energysum{
    energysum() {e15=e25=e50=e100=e250=e1000=0.0;}
    double e15, e25, e50, e100, e250, e1000;
  };

  struct HitsInfo{
    HitsInfo() {
      x=y=z=time=energy=phi=eta=0.0;
      layer=0;
    }
    float x, y, z, time, energy, phi, eta ;
    float layer;
  };

  virtual void beginJob() override {}
  virtual void endJob() override {}
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;

  template<class T1, class T2>
    void recHitValidation(DetId & detId, int layer, const T1* geom, T2 it);
  void fillHitsInfo(); 
  void fillHitsInfo(HitsInfo& hits); 
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  
  // ----------member data ---------------------------
  std::string           nameDetector_;
  edm::EDGetToken       recHitSource_;
  int                   verbosity_;
  bool                  ifHCAL_;
  unsigned int          layers_;
  std::map<int, int>    OccupancyMap_plus;
  std::map<int, int>    OccupancyMap_minus;

  std::vector<TH2D*>    EtaPhi_Plus_;
  std::vector<TH2D*>    EtaPhi_Minus_;
  std::vector<TH1D*>    energy_;
  std::vector<TH1D*>    HitOccupancy_Plus_;
  std::vector<TH1D*>    HitOccupancy_Minus_;
  TH1D                 *MeanHitOccupancy_Plus_, *MeanHitOccupancy_Minus_;
};


HGCalRecHitStudy::HGCalRecHitStudy(const edm::ParameterSet& iConfig) {

  usesResource("TFileService");
  layers_       = 0;
  nameDetector_ = iConfig.getParameter<std::string>("DetectorName");
  verbosity_    = iConfig.getUntrackedParameter<int>("Verbosity",0);
  auto temp     = iConfig.getParameter<edm::InputTag>("RecHitSource");
  ifHCAL_       = iConfig.getParameter<bool>("ifHCAL");
  if (nameDetector_ == "HGCalEESensitive" || 
      nameDetector_ == "HGCalHESiliconSensitive" ||
      nameDetector_ == "HGCalHEScintillatorSensitive" ) {
    recHitSource_    = consumes<HGCRecHitCollection>(temp);
  } else if ( nameDetector_ == "HCal" ) {
    if (ifHCAL_) recHitSource_ = consumes<HBHERecHitCollection>(temp);
    else         recHitSource_ = consumes<HGCRecHitCollection>(temp);
  } else {
    throw cms::Exception("BadHGCRecHitSource")
      << "HGCal DetectorName given as " << nameDetector_ << " must be: "
      << "\"HGCalHESiliconSensitive\", \"HGCalHESiliconSensitive\", "
      << "\"HGCalHEScintillatorSensitive\", or \"HCal\"!"; 
  }
  edm::LogVerbatim("HGCalValidation") << "Initialize HGCalRecHitStudy for " 
				      << nameDetector_  << " with i/p tag " 
				      << temp << " Flag " << ifHCAL_ << ":"
				      << verbosity_;
}


HGCalRecHitStudy::~HGCalRecHitStudy() { }

void HGCalRecHitStudy::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup) {
  OccupancyMap_plus.clear();
  OccupancyMap_minus.clear();

  bool ok(true);
  unsigned int ntot(0), nused(0);
  if (nameDetector_ == "HCal") {
    edm::ESHandle<CaloGeometry> geom;
    iSetup.get<CaloGeometryRecord>().get(geom);
    if (!geom.isValid()) 
      edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
    const CaloGeometry* geom0 = geom.product();

    if (ifHCAL_) {
      edm::Handle<HBHERecHitCollection> hbhecoll;
      iEvent.getByToken(recHitSource_, hbhecoll);
      if (hbhecoll.isValid()) {
	if (verbosity_>0) 
	  edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					      << hbhecoll->size() << " element(s)";
	for (auto it=hbhecoll->begin(); it != hbhecoll->end(); ++it) {
	  DetId detId = it->id();
	  ntot++;
	  if (detId.subdetId() == HcalEndcap) {
	    nused++;
	    int   layer = HcalDetId(detId).depth();
	    recHitValidation(detId, layer, geom0, it);
	  }
	}
      } else {
	ok = false;
	edm::LogWarning("HGCalValidation") << "HBHERecHitCollection Handle does not exist !!!";
      }
    } else {
      edm::Handle<HGCRecHitCollection> hbhecoll;
      iEvent.getByToken(recHitSource_, hbhecoll);
      if (hbhecoll.isValid()) {
	if (verbosity_>0) 
	  edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					      << hbhecoll->size() << " element(s)";
	for (auto it=hbhecoll->begin(); it != hbhecoll->end(); ++it) {
	  DetId detId = it->id();
	  ntot++; nused++;
	  int   layer = HcalDetId(detId).depth();
	  recHitValidation(detId, layer, geom0, it);
	}
      } else {
	ok = false;
	edm::LogWarning("HGCalValidation") << "HGChebRecHitCollection Handle does not exist !!!";
      }
    }
  } else {
    edm::ESHandle<HGCalGeometry> geom;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, geom);
    if (!geom.isValid()) edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
    const HGCalGeometry* geom0 = geom.product();

    edm::Handle<HGCRecHitCollection> theRecHitContainers;
    iEvent.getByToken(recHitSource_, theRecHitContainers);
    if (theRecHitContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					    << theRecHitContainers->size()
					    << " element(s)";
      for (auto it=theRecHitContainers->begin();
	   it !=theRecHitContainers->end(); ++it) {
	ntot++; nused++;
	DetId detId = it->id();
	int layer   = (detId.subdetId() == HGCEE) ? (HGCEEDetId(detId).layer()) : (HGCHEDetId(detId).layer());
	recHitValidation(detId, layer, geom0, it);
      }
    } else {
      ok = false;
      edm::LogWarning("HGCalValidation") << "HGCRecHitCollection Handle does not exist !!!";
    }
  }
  if (ok) fillHitsInfo();
  edm::LogWarning("HGCalValidation") << "Event " << iEvent.id().event()
				     << " with " << ntot << " total and "
				     << nused << " used recHits";
}

template<class T1, class T2>
void HGCalRecHitStudy::recHitValidation(DetId & detId, int layer, 
					const T1* geom, T2 it) {

  GlobalPoint global = geom->getPosition(detId);
  double      energy = it->energy();

  float globalx = global.x();
  float globaly = global.y();
  float globalz = global.z();
      
  HitsInfo   hinfo;
  hinfo.energy = energy;
  hinfo.x      = globalx;
  hinfo.y      = globaly;
  hinfo.z      = globalz;
  hinfo.layer  = layer;
  hinfo.phi    = global.phi();
  hinfo.eta    = global.eta();
      
  if (verbosity_>1) 
    edm::LogVerbatim("HGCalValidation")<< " --------------------------   gx = "
				       << globalx << " gy = "  << globaly   
				       << " gz = " << globalz << " phi = " 
				       << hinfo.phi << " eta = " << hinfo.eta;
      
  fillHitsInfo(hinfo);
      
  if (hinfo.eta > 0)  fillOccupancyMap(OccupancyMap_plus, layer -1);
  else                fillOccupancyMap(OccupancyMap_minus, layer -1);
      
}      

void HGCalRecHitStudy::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer){
  if (OccupancyMap.find(layer) != OccupancyMap.end()) OccupancyMap[layer]++;
  else                                                OccupancyMap[layer] = 1;
}

void HGCalRecHitStudy::fillHitsInfo() { 

  for (auto itr = OccupancyMap_plus.begin() ; itr != OccupancyMap_plus.end(); ++itr) {
    int layer      = (*itr).first;
    int occupancy  = (*itr).second;
    HitOccupancy_Plus_.at(layer)->Fill(occupancy);
  }

  for (auto itr = OccupancyMap_minus.begin() ; itr != OccupancyMap_minus.end(); ++itr) {
    int layer      = (*itr).first;
    int occupancy  = (*itr).second;
    HitOccupancy_Minus_.at(layer)->Fill(occupancy);
  }
  
}

void HGCalRecHitStudy::fillHitsInfo(HitsInfo& hits) {

  unsigned int ilayer = hits.layer -1;
  energy_.at(ilayer)->Fill(hits.energy);

  EtaPhi_Plus_.at(ilayer)->Fill(hits.eta , hits.phi);
  EtaPhi_Minus_.at(ilayer)->Fill(hits.eta, hits.phi);

}

void HGCalRecHitStudy::beginRun(edm::Run const&,
                                edm::EventSetup const& iSetup) {

  if (nameDetector_ == "HCal") {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
    const HcalDDDRecConstants *hcons   = &(*pHRNDC);
    layers_ = hcons->getMaxDepth(1);
  } else {
    edm::ESHandle<HGCalDDDConstants>  pHGDC;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, pHGDC);
    const HGCalDDDConstants & hgcons_ = (*pHGDC);
    layers_ = hgcons_.layers(true);
  }

  edm::LogVerbatim("HGCalValidation") << "Finds " << layers_ << " layers for " 
				      << nameDetector_;

  edm::Service<TFileService> fs;
  char histoname[100];
  for (unsigned int ilayer = 0; ilayer < layers_; ilayer++ ) {
    sprintf (histoname,"HitOccupancy_Plus_layer_%d", ilayer);
    HitOccupancy_Plus_.push_back(fs->make<TH1D>(histoname, "RecHitOccupancy_Plus", 100, 0, 10000));
    sprintf (histoname, "HitOccupancy_Minus_layer_%d", ilayer);
    HitOccupancy_Minus_.push_back(fs->make<TH1D>(histoname, "RecHitOccupancy_Minus", 100, 0, 10000));

    sprintf (histoname, "EtaPhi_Plus_Layer_%d", ilayer);
    EtaPhi_Plus_.push_back(fs->make<TH2D>(histoname, "Occupancy", 31, 1.45, 3.0, 72, -CLHEP::pi, CLHEP::pi));
    sprintf (histoname, "EtaPhi_Minus_Layer_%d", ilayer);
    EtaPhi_Minus_.push_back(fs->make<TH2D>(histoname, "Occupancy", 31, -3.0, -1.45, 72, -CLHEP::pi, CLHEP::pi));
      
    sprintf (histoname, "Energy_Layer_%d", ilayer); 
    energy_.push_back(fs->make<TH1D>(histoname,"Energy",1000,0,10.0));
  }//loop over layers ends here 

  MeanHitOccupancy_Plus_= fs->make<TH1D>("SUMOfRecHitOccupancy_Plus", "SUMOfRecHitOccupancy_Plus", layers_, -0.5, layers_-0.5);
  MeanHitOccupancy_Minus_ = fs->make<TH1D>("SUMOfRecHitOccupancy_Minus", "SUMOfRecHitOccupancy_Minus", layers_, -0.5,layers_-0.5);
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalRecHitStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("DetectorName","HGCalEESensitive");
  desc.add<edm::InputTag>("RecHitSource",edm::InputTag("HGCalRecHit","HGCEERecHits"));
  desc.add<bool>("ifHCAL",false);
  desc.addUntracked<int>("Verbosity",0);
  descriptions.add("hgcalRecHitStudyEE",desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalRecHitStudy);

