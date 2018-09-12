// system include files
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "TH2D.h"
#include "TH1D.h"

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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

class HGCalDigiStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit HGCalDigiStudy(const edm::ParameterSet&);
  ~HGCalDigiStudy() override {}
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:

  void beginJob() override;
  void endJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  struct digiInfo{
    digiInfo() {
      phi = eta = r = z = charge = 0.0;
      layer = adc = 0;
    }
    double phi, eta, r, z, charge;
    int layer, adc;
  };

  template<class T1, class T2> 
  void digiValidation(const T1& detId, const T2* geom, int indx, int layer, 
		      uint16_t adc, double charge);
  void fillDigiInfo(int indx, digiInfo& hinfo);
  
  // ----------member data ---------------------------
  const std::vector<std::string>        nameDetectors_;
  const bool                            ifNose_, ifHCAL_;
  const int                             verbosity_, SampleIndx_, nbinR_, nbinZ_;
  const double                          rmin_, rmax_, zmin_, zmax_;
  std::vector<edm::EDGetToken>          digiSources_;
  std::vector<const HGCalGeometry*>     hgcGeom_;
  const HcalGeometry*                   hcGeom_;
  std::vector<int>                      layers_, layerFront_, geomType_;

  std::vector<TH1D*>                    h_Charge_, h_ADC_, h_LayZp_, h_LayZm_;
  std::vector<TH2D*>                    h_RZ_, h_EtaPhi_;
};

HGCalDigiStudy::HGCalDigiStudy(const edm::ParameterSet& iConfig) :
  nameDetectors_(iConfig.getParameter<std::vector<std::string> >("detectorNames")),
  ifNose_(iConfig.getUntrackedParameter<bool>("ifNose",false)),
  ifHCAL_(iConfig.getUntrackedParameter<bool>("ifHCAL",false)),
  verbosity_(iConfig.getUntrackedParameter<int>("verbosity",0)),
  SampleIndx_(iConfig.getUntrackedParameter<int>("sampleIndex",5)),
  nbinR_(iConfig.getUntrackedParameter<int>("nBinR",300)),
  nbinZ_(iConfig.getUntrackedParameter<int>("nBinZ",300)),
  rmin_(iConfig.getUntrackedParameter<double>("rMin",0.0)),
  rmax_(iConfig.getUntrackedParameter<double>("rMax",3000.0)),
  zmin_(iConfig.getUntrackedParameter<double>("zMin",3000.0)),
  zmax_(iConfig.getUntrackedParameter<double>("zMax",6000.0)),
  hcGeom_(nullptr) {

  usesResource(TFileService::kSharedResource);

  auto temp = iConfig.getParameter<std::vector<edm::InputTag> >("DigiSources");
  for (unsigned int k=0; k<temp.size(); ++k) {
    if ((nameDetectors_[k] == "HGCalEESensitive") || 
	(nameDetectors_[k] == "HGCalHESiliconSensitive") || 
	(nameDetectors_[k] == "HGCalHEScintillatorSensitive") ||
	(nameDetectors_[k] == "HFNoseSensitive")) {
      digiSources_.emplace_back(consumes<HGCalDigiCollection>(temp[k]));
    } else if (nameDetectors_[k] == "HCal") {
      if (ifHCAL_) 
	digiSources_.emplace_back(consumes<QIE11DigiCollection>(temp[k]));
      else 
        digiSources_.emplace_back(consumes<HGCalDigiCollection>(temp[k]));
    } else {
      throw cms::Exception("BadHGCDigiSource")
	<< "HGCal DetectorName given as " << nameDetectors_[k] << " must be: "
	<< "\"HGCalEESensitive\", \"HGCalHESiliconSensitive\", "
	<< "\"HGCalHEScintillatorSensitive\", \"HFNoseSensitive\", "
	<< "or \"HCal\"!"; 
    }
    edm::LogVerbatim("HGCalValidation") << "HGCalDigiStudy: request for Digi "
					<< "collection " << temp[k] << " for "
					<< nameDetectors_[k];
  }  
}

void HGCalDigiStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string>   names   = {"HGCalEESensitive",
					"HGCalHESiliconSensitive",
					"Hcal"};
  std::vector<edm::InputTag> sources = {edm::InputTag("simHGCalUnsuppressedlDigis","EE"),
					edm::InputTag("simHGCalUnsuppressedlDigis","HEfront"),
					edm::InputTag("simHGCalUnsuppressedlDigis","HEback")};
  desc.add<std::vector<std::string> >("detectorNames",names);
  desc.add<std::vector<edm::InputTag> >("DigiSources",sources);
  desc.addUntracked<bool>("ifNose",false);
  desc.addUntracked<bool>("ifHCAL",false);
  desc.addUntracked<int>("verbosity",0);
  desc.addUntracked<int>("sampleIndex",0);
  desc.addUntracked<double>("rMin",0.0);
  desc.addUntracked<double>("rMax",3000.0);
  desc.addUntracked<double>("zMin",3000.0);
  desc.addUntracked<double>("zMax",6000.0);
  desc.addUntracked<int>("nBinR",300);
  desc.addUntracked<int>("nBinZ",300);
  descriptions.add("hgcalDigiStudy",desc);
}

void HGCalDigiStudy::beginJob() {

  edm::Service<TFileService> fs;
  std::ostringstream hname, title;
  for (auto const& name : nameDetectors_) {
    hname.str(""); title.str("");
    hname << "RZ_" << name;
    title << "R vs Z for " << name;
    h_RZ_.emplace_back(fs->make<TH2D>(hname.str().c_str(), 
                                      title.str().c_str(),
                                      nbinZ_,zmin_,zmax_,nbinR_,rmin_,rmax_));
    hname.str(""); title.str("");
    hname << "EtaPhi_" << name;
    title << "#phi vs #eta for " << name;
    h_EtaPhi_.emplace_back(fs->make<TH2D>(hname.str().c_str(), 
                                          title.str().c_str(),
                                          200,1.0,3.0,200,-M_PI,M_PI));
    hname.str(""); title.str("");
    hname << "Charge_" << name;
    title << "Charge for " << name;
    h_Charge_.emplace_back(fs->make<TH1D>(hname.str().c_str(),
                                          title.str().c_str(),100,-25,25));
    hname.str(""); title.str("");
    hname << "ADC_" << name;
    title << "ADC for " << name;
    h_ADC_.emplace_back(fs->make<TH1D>(hname.str().c_str(),
				       title.str().c_str(),200,0,1000));
    hname.str(""); title.str("");
    hname  << "LayerZp_" << name;
    title << "Charge vs Layer (+z) for " << name;
    h_LayZp_.emplace_back(fs->make<TH1D>(hname.str().c_str(), 
					 title.str().c_str(),60,0.0,60.0));
    hname.str(""); title.str("");
    hname  << "LayerZm_" << name;
    title << "Charge vs Layer (-z) for " << name;
    h_LayZm_.emplace_back(fs->make<TH1D>(hname.str().c_str(), 
					 title.str().c_str(),60,0.0,60.0));
  }
}

void HGCalDigiStudy::beginRun(const edm::Run&, 
                                const edm::EventSetup& iSetup) {

  for (const auto& name : nameDetectors_) {
    int type(0), layers(0), layerfront(0);
    if (name == "HCal") {
      edm::ESHandle<CaloGeometry> geom;
      iSetup.get<CaloGeometryRecord>().get(geom);
      if (!geom.isValid()) 
	edm::LogVerbatim("HGCalValidation") << "HGCalDigiStudy: Cannot get "
					    << "valid Geometry Object for " 
					    << name;
      hcGeom_    = static_cast<const HcalGeometry*>
	((geom.product())->getSubdetectorGeometry(DetId::Hcal,HcalBarrel));
      hgcGeom_.push_back(nullptr);
      layers     = hcGeom_->topology().maxDepthHE();
      layerfront = 40;
    } else {
      edm::ESHandle<HGCalGeometry> geom;
      iSetup.get<IdealGeometryRecord>().get(name, geom);
      if (!geom.isValid()) 
	edm::LogVerbatim("HGCalValidation") << "HGCalDigiStudy: Cannot get "
					    << "valid Geometry Object for "
					    << name;
      const HGCalGeometry* geom0 = geom.product();
      HGCalGeometryMode::GeometryMode mode = geom0->topology().geomMode();
      if ((mode == HGCalGeometryMode::Hexagon8) ||
	  (mode == HGCalGeometryMode::Hexagon8Full)) type = 1;
      else if (mode == HGCalGeometryMode::Trapezoid) type = 2;
      if (name == "HFNoseSensitive") {
	type = 3;
      } else if (name != "HGCalEESensitive") {
	layerfront = 28;
      }
      hgcGeom_.emplace_back(geom0);
      layers    = geom0->topology().dddConstants().layers(true);
    }
    layers_.emplace_back(layers);
    layerFront_.emplace_back(layerfront);
    geomType_.emplace_back(type);
    edm::LogVerbatim("HGCalValidation") << "HGCalDigiStudy: gets Geometry for "
					<< name << " of type " << type
					<< " with " << layers << " layers and "
					<< "front layer " << layerfront;
  }
}

void HGCalDigiStudy::analyze(const edm::Event& iEvent, 
			     const edm::EventSetup& iSetup) { 

  for (unsigned int k=0; k<nameDetectors_.size(); ++k) {
    unsigned int ntot(0), nused(0);

    if ((nameDetectors_[k] == "HGCalEESensitive") ||
	(nameDetectors_[k] == "HFNoseSensitive")) {
      //HGCalEE
      edm::Handle<HGCalDigiCollection> theHGCEEDigiContainer;
      iEvent.getByToken(digiSources_[k], theHGCEEDigiContainer);
      if (theHGCEEDigiContainer.isValid()) {
	if (verbosity_>0) 
	  edm::LogVerbatim("HGCalValidation") << nameDetectors_[k] << " with " 
					      << theHGCEEDigiContainer->size() 
					      << " element(s)";
	for (const auto & it: *(theHGCEEDigiContainer.product())) {
	  ntot++; nused++;
	  DetId      detId     = it.id();
	  int        layer     = ((geomType_[k] == 0) ? 
				  (HGCalDetId(detId).layer()) :
				  (geomType_[k] == 1) ?
				  HGCSiliconDetId(detId).layer() :
				  HFNoseDetId(detId).layer());
	  const HGCSample&  hgcSample = it.sample(SampleIndx_);
	  uint16_t   gain      = hgcSample.toa();
	  uint16_t   adc       = hgcSample.data();
	  double     charge    = adc*gain;
	  digiValidation(detId, hgcGeom_[k], k, layer, adc, charge);
	}
      } else {
	edm::LogVerbatim("HGCalValidation") << "DigiCollection handle does not "
					    << "exist for " 
					    << nameDetectors_[k] << " !!!";
      }
    } else if ((nameDetectors_[k] == "HGCalHESiliconSensitive") || 
	       (nameDetectors_[k] == "HGCalHEScintillatorSensitive")) {
      //HGCalHE
      edm::Handle<HGCalDigiCollection> theHGCHEDigiContainer;
      iEvent.getByToken(digiSources_[k], theHGCHEDigiContainer);
      if (theHGCHEDigiContainer.isValid()) {
	if (verbosity_>0) 
	  edm::LogVerbatim("HGCalValidation") << nameDetectors_[k] << " with " 
					      << theHGCHEDigiContainer->size()
					      << " element(s)";
	for (const auto & it: *(theHGCHEDigiContainer.product())) {
	  ntot++; nused++;
	  DetId      detId     = it.id();
	  int        layer     = ((geomType_[k] == 0) ? 
				  HGCalDetId(detId).layer() :
				  ((geomType_[k] == 1) ? 
				   HGCSiliconDetId(detId).layer() :
				   HGCScintillatorDetId(detId).layer()));
	  const HGCSample&  hgcSample = it.sample(SampleIndx_);
	  uint16_t   gain      = hgcSample.toa();
	  uint16_t   adc       = hgcSample.data();
	  double     charge    = adc*gain;
	  digiValidation(detId, hgcGeom_[k], k, layer, adc, charge);
	}
      } else {
	edm::LogVerbatim("HGCalValidation") << "DigiCollection handle does not "
					    << "exist for " 
					    << nameDetectors_[k] << " !!!";
      }
    } else if ((nameDetectors_[k] == "HCal") && (!ifHCAL_)) {
      //HGCalBH
      edm::Handle<HGCalDigiCollection> theHGCBHDigiContainer;
      iEvent.getByToken(digiSources_[k], theHGCBHDigiContainer);
      if (theHGCBHDigiContainer.isValid()) {
	if (verbosity_>0) 
	  edm::LogVerbatim("HGCalValidation") << nameDetectors_[k] << " with " 
					      << theHGCBHDigiContainer->size()
					      << " element(s)";
	for (const auto & it: *(theHGCBHDigiContainer.product())) {
	  ntot++; nused++;
	  HcalDetId  detId     = it.id();
	  int        layer     = detId.depth();
	  const HGCSample&  hgcSample = it.sample(SampleIndx_);
	  uint16_t   gain      = hgcSample.toa();
	  uint16_t   adc       = hgcSample.data();
	  double     charge    = adc*gain;
	  digiValidation(detId, hcGeom_, k, layer, adc, charge);
	}
      } else {
	edm::LogWarning("HGCalValidation") << "DigiCollection handle does not "
					   << "exist for "
					   << nameDetectors_[k] << " !!!";
      }
    } else if (nameDetectors_[k] == "HCal") {
      //HE
      edm::Handle<QIE11DigiCollection>  theHEDigiContainer;
      iEvent.getByToken(digiSources_[k], theHEDigiContainer);
      if (theHEDigiContainer.isValid()) {
	if (verbosity_>0) 
	  edm::LogVerbatim("HGCalValidation") << nameDetectors_[k] << " with " 
					      << theHEDigiContainer->size()
					      << " element(s)";
	edm::ESHandle<HcalDbService> conditions;
	iSetup.get<HcalDbRecord > ().get(conditions);

	for (const auto & it: *(theHEDigiContainer.product())) {
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
	    digiValidation(detId, hcGeom_, k, layer, adc, charge);
	  }
	}
      } else {
	edm::LogWarning("HGCalValidation") << "DigiCollection handle does not "
					   << "exist for "
					   << nameDetectors_[k] << " !!!";
      }
    } else {
      edm::LogWarning("HGCalValidation") << "invalid detector name !! " 
					 << nameDetectors_[k];
    }
    edm::LogVerbatim("HGCalValidation") << "Event " << iEvent.id().event()
					<< ":" << nameDetectors_[k]
					<< " with " << ntot << " total and "
					<< nused << " used digis";
  }
}

template<class T1, class T2>
void HGCalDigiStudy::digiValidation(const T1& detId, const T2* geom, 
				    int indx, int layer, uint16_t adc, 
				    double charge) {
  
  if (verbosity_>1) 
    edm::LogVerbatim("HGCalValidation") << std::hex 
					<< detId.rawId() << std::dec
					<< " adc = "         <<  adc
					<< " charge = "      <<  charge;

  DetId id1 = DetId(detId.rawId());
  const GlobalPoint& gcoord = geom->getPosition(id1);
  
  digiInfo   hinfo; 
  hinfo.r       =  gcoord.perp();
  hinfo.z       =  gcoord.z();
  hinfo.eta     =  std::abs(gcoord.eta());
  hinfo.phi     =  gcoord.phi();
  hinfo.adc     =  adc;
  hinfo.charge  =  charge;
  hinfo.layer   =  layer+layerFront_[indx];
  if (verbosity_>1) 
    edm::LogVerbatim("HGCalValidation") << "R =  "  << hinfo.r
					<< " z = "  << hinfo.z
					<< " eta = "  << hinfo.eta
					<< " phi = "  << hinfo.phi;

  fillDigiInfo(indx, hinfo);
  
}

void HGCalDigiStudy::fillDigiInfo(int indx, digiInfo& hinfo) {
  h_Charge_[indx]->Fill(hinfo.charge);
  h_ADC_[indx]->Fill(hinfo.adc);
  h_RZ_[indx]->Fill(std::abs(hinfo.z),hinfo.r);
  h_EtaPhi_[indx]->Fill(hinfo.eta,hinfo.phi);
  if (hinfo.z > 0) h_LayZp_[indx]->Fill(hinfo.layer,hinfo.charge);
  else             h_LayZm_[indx]->Fill(hinfo.layer,hinfo.charge);
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCalDigiStudy);
