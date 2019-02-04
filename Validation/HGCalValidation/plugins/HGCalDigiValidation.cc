// system include files
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

// user include files


class HGCalDigiValidation : public DQMEDAnalyzer {

public:
  struct digiInfo{
    digiInfo() {
      x = y = z = charge = 0.0;
      layer = adc = 0;
    }
    double x, y, z, charge;
    int layer, adc;
  };

  explicit HGCalDigiValidation(const edm::ParameterSet&);
  ~HGCalDigiValidation() override {}
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void fillDigiInfo(digiInfo&   hinfo);
  void fillDigiInfo();
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  template<class T1, class T2> 
  void digiValidation(const T1& detId, const T2* geom, int layer, 
		      uint16_t adc, double charge);
  
  // ----------member data ---------------------------
  std::string       nameDetector_;
  edm::EDGetToken   digiSource_;
  bool              ifNose_, ifHCAL_;
  int               verbosity_, SampleIndx_;
  int               layers_, firstLayer_;
 
  std::map<int, int> OccupancyMap_plus_;
  std::map<int, int> OccupancyMap_minus_;

  std::vector<MonitorElement*> charge_;
  std::vector<MonitorElement*> DigiOccupancy_XY_;
  std::vector<MonitorElement*> ADC_;
  std::vector<MonitorElement*> DigiOccupancy_Plus_;
  std::vector<MonitorElement*> DigiOccupancy_Minus_;
  MonitorElement* MeanDigiOccupancy_Plus_;
  MonitorElement* MeanDigiOccupancy_Minus_;
};

HGCalDigiValidation::HGCalDigiValidation(const edm::ParameterSet& iConfig) :
  nameDetector_(iConfig.getParameter<std::string>("DetectorName")),
  ifNose_(iConfig.getParameter<bool>("ifNose")),
  ifHCAL_(iConfig.getParameter<bool>("ifHCAL")),
  verbosity_(iConfig.getUntrackedParameter<int>("Verbosity",0)),
  SampleIndx_(iConfig.getUntrackedParameter<int>("SampleIndx",0)),
  firstLayer_(1) {

  auto temp = iConfig.getParameter<edm::InputTag>("DigiSource");
  if ((nameDetector_ == "HGCalEESensitive") || 
      (nameDetector_ == "HGCalHESiliconSensitive") || 
      (nameDetector_ == "HGCalHEScintillatorSensitive") ||
      (nameDetector_ == "HGCalHFNoseSensitive")) {
    digiSource_    = consumes<HGCalDigiCollection>(temp);
  } else if (nameDetector_ == "HCal") {
    if (ifHCAL_) digiSource_ = consumes<QIE11DigiCollection>(temp);
    else         digiSource_ = consumes<HGCalDigiCollection>(temp);
  } else {
    throw cms::Exception("BadHGCDigiSource")
      << "HGCal DetectorName given as " << nameDetector_ << " must be: "
      << "\"HGCalEESensitive\", \"HGCalHESiliconSensitive\", "
      << "\"HGCalHEScintillatorSensitive\", \"HGCalHFNoseSensitive\", "
      << "or \"HCal\"!"; 
  }  
}

void HGCalDigiValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("DetectorName","HGCalEESensitive");
  desc.add<edm::InputTag>("DigiSource",edm::InputTag("hgcalDigis","EE"));
  desc.add<bool>("ifNose",false);
  desc.add<bool>("ifHCAL",false);
  desc.addUntracked<int>("Verbosity",0);
  desc.addUntracked<int>("SampleIndx",0);
  descriptions.add("hgcalDigiValidationEEDefault",desc);
}

void HGCalDigiValidation::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) { 
  OccupancyMap_plus_.clear();
  OccupancyMap_minus_.clear();
  
  const HGCalGeometry* geom0(nullptr);
  const CaloGeometry*  geom1(nullptr);
  int geomType(0);
  if (nameDetector_ == "HCal") {
    edm::ESHandle<CaloGeometry> geom;
    iSetup.get<CaloGeometryRecord>().get(geom);
    if (!geom.isValid()) 
      edm::LogVerbatim("HGCalValidation") << "HGCalDigiValidation: Cannot get "
					  << "valid Geometry Object for " 
					  << nameDetector_;
    geom1 = geom.product();
  } else {
    edm::ESHandle<HGCalGeometry> geom;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, geom);
    if (!geom.isValid()) 
      edm::LogVerbatim("HGCalValidation") << "HGCalDigiValidation: Cannot get "
					  << "valid Geometry Object for " 
					  << nameDetector_;
    geom0 = geom.product();
    HGCalGeometryMode::GeometryMode mode = geom0->topology().geomMode();
    if ((mode == HGCalGeometryMode::Hexagon8) ||
	(mode == HGCalGeometryMode::Hexagon8Full)) geomType = 1;
    else if (mode == HGCalGeometryMode::Trapezoid) geomType = 2;
    if (nameDetector_ == "HGCalHFNoseSensitive")   geomType = 3;
  }

  unsigned int ntot(0), nused(0);
  if ((nameDetector_ == "HGCalEESensitive") ||
      (nameDetector_ == "HGCalHFNoseSensitive")) {
    //HGCalEE
    edm::Handle<HGCalDigiCollection> theHGCEEDigiContainers;
    iEvent.getByToken(digiSource_, theHGCEEDigiContainers);
    if (theHGCEEDigiContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					    << theHGCEEDigiContainers->size() 
					    << " element(s)";
      for (const auto & it: *(theHGCEEDigiContainers.product())) {
	ntot++; nused++;
	DetId      detId     = it.id();
	int        layer     = ((geomType == 0) ? HGCalDetId(detId).layer() :
				(geomType == 1) ?
				HGCSiliconDetId(detId).layer() :
				HFNoseDetId(detId).layer());
	const HGCSample&  hgcSample = it.sample(SampleIndx_);
	uint16_t   gain      = hgcSample.toa();
	uint16_t   adc       = hgcSample.data();
	double     charge    = adc*gain;
	digiValidation(detId, geom0, layer, adc, charge);
      }
      fillDigiInfo();
    } else {
      edm::LogVerbatim("HGCalValidation") << "DigiCollection handle does not "
					  << "exist for " << nameDetector_;
    }
  } else if ((nameDetector_ == "HGCalHESiliconSensitive") || 
	     (nameDetector_ == "HGCalHEScintillatorSensitive")) {
    //HGCalHE
    edm::Handle<HGCalDigiCollection> theHGCHEDigiContainers;
    iEvent.getByToken(digiSource_, theHGCHEDigiContainers);
    if (theHGCHEDigiContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					    << theHGCHEDigiContainers->size()
					    << " element(s)";
      for (const auto & it: *(theHGCHEDigiContainers.product())) {
	ntot++; nused++;
	DetId      detId     = it.id();
	int        layer     = ((geomType == 0) ? HGCalDetId(detId).layer() :
				((geomType == 1) ? HGCSiliconDetId(detId).layer() :
				 HGCScintillatorDetId(detId).layer()));
	const HGCSample&  hgcSample = it.sample(SampleIndx_);
	uint16_t   gain      = hgcSample.toa();
	uint16_t   adc       = hgcSample.data();
	double     charge    = adc*gain;
	digiValidation(detId, geom0, layer, adc, charge);
      }
      fillDigiInfo();
    } else {
      edm::LogVerbatim("HGCalValidation") << "DigiCollection handle does not "
					  << "exist for " << nameDetector_;
    }
  } else if ((nameDetector_ == "HCal") && (!ifHCAL_)) {
    //HGCalBH
    edm::Handle<HGCalDigiCollection> theHGCBHDigiContainers;
    iEvent.getByToken(digiSource_, theHGCBHDigiContainers);
    if (theHGCBHDigiContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					    << theHGCBHDigiContainers->size()
					    << " element(s)";
      for (const auto & it: *(theHGCBHDigiContainers.product())) {
	ntot++; nused++;
	HcalDetId  detId     = it.id();
	int        layer     = detId.depth();
	const HGCSample&  hgcSample = it.sample(SampleIndx_);
	uint16_t   gain      = hgcSample.toa();
	uint16_t   adc       = hgcSample.data();
	double     charge    = adc*gain;
	digiValidation(detId, geom1, layer, adc, charge);
      }
      fillDigiInfo();
    } else {
      edm::LogWarning("HGCalValidation") << "DigiCollection handle does not "
					 << "exist for " << nameDetector_;
    }
  } else if (nameDetector_ == "HCal") {
    //HE
    edm::Handle<QIE11DigiCollection>  theHEDigiContainers;
    iEvent.getByToken(digiSource_, theHEDigiContainers);
    if (theHEDigiContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					    << theHEDigiContainers->size()
					    << " element(s)";
      edm::ESHandle<HcalDbService> conditions;
      iSetup.get<HcalDbRecord > ().get(conditions);

      for (const auto & it: *(theHEDigiContainers.product())) {
	QIE11DataFrame df(it);
	HcalDetId detId  = (df.id());
	ntot++;
	if (detId.subdet() == HcalEndcap) {
	  nused++;
	  HcalCalibrations calibrations = conditions->getHcalCalibrations(detId);
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder(detId);
	  const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
	  HcalCoderDb coder(*channelCoder, *shape);
	  CaloSamples tool;
	  coder.adc2fC(df, tool);
	  int       layer  = detId.depth();
	  uint16_t  adc    = (df)[SampleIndx_].adc();
	  int       capid  = (df)[SampleIndx_].capid();
	  double    charge = (tool[SampleIndx_] - calibrations.pedestal(capid));
	  digiValidation(detId, geom1, layer, adc, charge);
	}
      }
      fillDigiInfo();
    } else {
      edm::LogWarning("HGCalValidation") << "DigiCollection handle does not "
					 << "exist for " << nameDetector_;
    }
  } else {
    edm::LogWarning("HGCalValidation") << "invalid detector name !! " 
				       << nameDetector_;
  }
  edm::LogVerbatim("HGCalValidation") << "Event " << iEvent.id().event()
				      << " with " << ntot << " total and "
				      << nused << " used digis";
}

template<class T1, class T2>
void HGCalDigiValidation::digiValidation(const T1& detId, const T2* geom, 
					 int layer, uint16_t adc, 
					 double charge) {
  
  if (verbosity_>1) edm::LogVerbatim("HGCalValidation") << std::hex 
							<< detId.rawId()
							<< std::dec;
  DetId id1 = DetId(detId.rawId());
  const GlobalPoint& global1 = geom->getPosition(id1);
  
  if (verbosity_>1) 
    edm::LogVerbatim("HGCalValidation") << " adc = "         <<  adc
					<< " charge = "      <<  charge;
  
  digiInfo   hinfo; 
  hinfo.x       =  global1.x();
  hinfo.y       =  global1.y();
  hinfo.z       =  global1.z();
  hinfo.adc     =  adc;
  hinfo.charge  =  charge;
  hinfo.layer   =  layer-firstLayer_;
  
  if (verbosity_>1) 
    edm::LogVerbatim("HGCalValidation") << "gx =  "  << hinfo.x
					<< " gy = "  << hinfo.y
					<< " gz = "  << hinfo.z;
  
  fillDigiInfo(hinfo);

  if (global1.eta() > 0)  fillOccupancyMap(OccupancyMap_plus_,  hinfo.layer);
  else                    fillOccupancyMap(OccupancyMap_minus_, hinfo.layer);
  
}

void HGCalDigiValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, 
					   int layer) {
  if (OccupancyMap.find(layer) != OccupancyMap.end()) OccupancyMap[layer] ++;
  else                                                OccupancyMap[layer] = 1;
}

void HGCalDigiValidation::fillDigiInfo(digiInfo& hinfo) {
  int ilayer = hinfo.layer;
  charge_.at(ilayer)->Fill(hinfo.charge);
  DigiOccupancy_XY_.at(ilayer)->Fill(hinfo.x, hinfo.y);
  ADC_.at(ilayer)->Fill(hinfo.adc);
}

void HGCalDigiValidation::fillDigiInfo() {
  for (const auto & itr : OccupancyMap_plus_) {
    int layer     = itr.first;
    int occupancy = itr.second;
    DigiOccupancy_Plus_.at(layer)->Fill(occupancy);
  }
  for (const auto & itr : OccupancyMap_minus_) {
    int layer     = itr.first;
    int occupancy = itr.second;
    DigiOccupancy_Minus_.at(layer)->Fill(occupancy);
  }
}

void HGCalDigiValidation::dqmBeginRun(const edm::Run&, 
				      const edm::EventSetup& iSetup) {
  if (nameDetector_ == "HCal") {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
    const HcalDDDRecConstants *hcons   = &(*pHRNDC);
    layers_ = hcons->getMaxDepth(1);
  } else {
    edm::ESHandle<HGCalDDDConstants>  pHGDC;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, pHGDC);
    const HGCalDDDConstants & hgcons_ = (*pHGDC);
    layers_     = hgcons_.layers(true);
    firstLayer_ = hgcons_.firstLayer();
  }
  
  if (verbosity_>0) 
    edm::LogVerbatim("HGCalValidation") << "current DQM directory:  "
					<< "HGCAL/HGCalDigisV/" 
					<< nameDetector_ << "  layer = "
					<< layers_ << " with the first one at "
					<< firstLayer_;
}  

void HGCalDigiValidation::bookHistograms(DQMStore::IBooker& iB, 
					 edm::Run const&, 
					 edm::EventSetup const&) {
  
  iB.setCurrentFolder("HGCAL/HGCalDigisV/"+nameDetector_);

  std::ostringstream histoname;
  for (int il = 0; il < layers_; ++il) {
    int ilayer = firstLayer_ + il;
    histoname.str(""); histoname << "charge_"<< "layer_" << ilayer;
    charge_.push_back(iB.book1D(histoname.str().c_str(),"charge_",100,-25,25));
      
    histoname.str(""); histoname << "ADC_" << "layer_" << ilayer;
    ADC_.push_back(iB.book1D(histoname.str().c_str(), "DigiOccupancy",200,0,1000));
      
    histoname.str(""); histoname << "DigiOccupancy_XY_" << "layer_" << ilayer;
    DigiOccupancy_XY_.push_back(iB.book2D(histoname.str().c_str(), "DigiOccupancy", 50, -500, 500, 50, -500, 500));
      
    histoname.str(""); histoname << "DigiOccupancy_Plus_" << "layer_" << ilayer;
    DigiOccupancy_Plus_.push_back(iB.book1D(histoname.str().c_str(), "DigiOccupancy +z", 100, 0, 1000));
    histoname.str(""); histoname << "DigiOccupancy_Minus_" << "layer_" << ilayer;
    DigiOccupancy_Minus_.push_back(iB.book1D(histoname.str().c_str(), "DigiOccupancy -z", 100, 0, 1000));
  }

  histoname.str(""); histoname << "SUMOfDigiOccupancy_Plus";
  MeanDigiOccupancy_Plus_ = iB.book1D( histoname.str().c_str(), "SUMOfDigiOccupancy_Plus", layers_, -0.5, layers_-0.5);
  histoname.str(""); histoname << "SUMOfRecDigiOccupancy_Minus";
  MeanDigiOccupancy_Minus_ = iB.book1D( histoname.str().c_str(), "SUMOfDigiOccupancy_Minus", layers_, -0.5,layers_-0.5);
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalDigiValidation);
