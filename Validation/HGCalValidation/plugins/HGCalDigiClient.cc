#include "Validation/HGCalValidation/plugins/HGCalDigiClient.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

HGCalDigiClient::HGCalDigiClient(const edm::ParameterSet& iConfig) {

  dbe_ = edm::Service<DQMStore>().operator->();
  nameDetector_  = iConfig.getParameter<std::string>("DetectorName");
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  verbosity_     = iConfig.getUntrackedParameter<int>("Verbosity",0);

  if (!dbe_) {
    edm::LogError("HGCalValidation") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if (iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if (dbe_) dbe_->setVerbose(0);
  }
  if (dbe_) dbe_->setCurrentFolder("/");
}

HGCalDigiClient::~HGCalDigiClient() { }

void HGCalDigiClient::beginJob() { }

void HGCalDigiClient::endJob() {
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void HGCalDigiClient::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) { 
  
  if (nameDetector_ == "HCal") {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
    const HcalDDDRecConstants *hcons  = &(*pHRNDC);
    layers_ = hcons->getMaxDepth(1);
  } else {
    edm::ESTransientHandle<DDCompactView> pDD;
    iSetup.get<IdealGeometryRecord>().get( pDD );
    const DDCompactView & cview = *pDD;
    HGCalDDDConstants *hgcons_ = new HGCalDDDConstants(cview, nameDetector_);
    layers_ = hgcons_->layers(true);
  }
}

void HGCalDigiClient::endRun(const edm::Run& , const edm::EventSetup& ) {
  runClient_();
}

//dummy analysis function
void HGCalDigiClient::analyze(const edm::Event& , const edm::EventSetup&) { }
void HGCalDigiClient::endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup& ) { }

void HGCalDigiClient::runClient_() {
  if (!dbe_) return; 
  dbe_->setCurrentFolder("/"); 
  if (verbosity_) edm::LogInfo("HGCalValidation") << "\nrunClient";
  std::vector<MonitorElement*> hgcalMEs;
  std::vector<std::string> fullDirPath = dbe_->getSubdirs();

  for (unsigned int i=0; i<fullDirPath.size(); i++) {
    if (verbosity_) 
      edm::LogInfo("HGCalValidation") << "\nfullPath: " << fullDirPath.at(i);
    dbe_->setCurrentFolder(fullDirPath.at(i));
    std::vector<std::string> fullSubDirPath = dbe_->getSubdirs();

    for (unsigned int j=0; j<fullSubDirPath.size(); j++) {
      if (verbosity_) 
	edm::LogInfo("HGCalValidation") << "fullSubPath: " << fullSubDirPath.at(j);
      std::string nameDirectory = "HGCalDigiV/"+nameDetector_;
      if (strcmp(fullSubDirPath.at(j).c_str(), nameDirectory.c_str()) == 0) {
        hgcalMEs = dbe_->getContents(fullSubDirPath.at(j));
        if (verbosity_) 
	  edm::LogInfo("HGCalValidation") << "hgcalMES size : " << hgcalMEs.size();
        if (!digisEndjob(hgcalMEs)) 
	  edm::LogInfo("HGCalValidation") << "\nError in DigisEndjob!";
      }
    }

  }
}

int HGCalDigiClient::digisEndjob(const std::vector<MonitorElement*>& hgcalMEs) {

  std::vector<MonitorElement*> charge_;
  std::vector<MonitorElement*> DigiOccupancy_XY_;
  std::vector<MonitorElement*> ADC_;
  std::vector<MonitorElement*> DigiOccupancy_Plus_;
  std::vector<MonitorElement*> DigiOccupancy_Minus_;
  std::vector<MonitorElement*> MeanDigiOccupancy_Plus_;
  std::vector<MonitorElement*> MeanDigiOccupancy_Minus_;
  std::ostringstream name;
  double nevent;
  int nbinsx, nbinsy;
  
  for (int ilayer = 0; ilayer < layers_; ilayer++ ){ 
    //charge
    name.str(""); name << "charge_layer_" << ilayer;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0)
	charge_.push_back(hgcalMEs[ih]);
    }
    //normalization
    nevent = charge_.at(ilayer)->getEntries();
    nbinsx = charge_.at(ilayer)->getNbinsX();
    for(int i=1; i <= nbinsx; i++) {
      double binValue = charge_.at(ilayer)->getBinContent(i)/nevent;
      charge_.at(ilayer)->setBinContent(i,binValue);
    }
    
    //XY 2d plots
    name.str(""); name << "DigiOccupancy_XY_" << "layer_" << ilayer;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0)
	DigiOccupancy_XY_.push_back(hgcalMEs[ih]);
    }
    
    //normalization of XY 2d
    nevent = DigiOccupancy_XY_.at(ilayer)->getEntries();
    nbinsx = DigiOccupancy_XY_.at(ilayer)->getNbinsX();
    nbinsy = DigiOccupancy_XY_.at(ilayer)->getNbinsY();
    for(int i=1; i<= nbinsx; ++i) {
      for(int j=1; j<= nbinsy; ++j) {
	double binValue = DigiOccupancy_XY_.at(ilayer)->getBinContent(i, j)/nevent;
	DigiOccupancy_XY_.at(ilayer)->setBinContent(i, j, binValue);
      }
    }
    
    //ADC
    name.str(""); name << "ADC_layer_" << ilayer;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0)
	ADC_.push_back(hgcalMEs[ih]);
    }
    
    //normalization of ADC Histos
    nevent = ADC_.at(ilayer)->getEntries();
    nbinsx = ADC_.at(ilayer)->getNbinsX();
    for(int i=1; i<= nbinsx; ++i) {
      double binValue = ADC_.at(ilayer)->getBinContent(i)/nevent;
      ADC_.at(ilayer)->setBinContent(i, binValue);
    }
    
    //Digi Occupancy
    name.str(""); name << "DigiOccupancy_Plus_layer_" << ilayer;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
        DigiOccupancy_Plus_.push_back(hgcalMEs[ih]);
      }
    }
    
    name.str(""); name << "DigiOccupancy_Minus_layer_" << ilayer;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
        DigiOccupancy_Minus_.push_back(hgcalMEs[ih]);
      }
    }
    
    //normalization of Digi Occupancy histos
    nevent = DigiOccupancy_Plus_.at(ilayer)->getEntries();
    nbinsx = DigiOccupancy_Plus_.at(ilayer)->getNbinsX();
    for(int i=1; i<= nbinsx; ++i) {
      double binValue = DigiOccupancy_Plus_.at(ilayer)->getBinContent(i)/nevent;
      DigiOccupancy_Plus_.at(ilayer)->setBinContent(i, binValue);
    }
    
    nevent = DigiOccupancy_Minus_.at(ilayer)->getEntries();
    nbinsx = DigiOccupancy_Minus_.at(ilayer)->getNbinsX();
    for(int i=1; i<= nbinsx; ++i) {
      double binValue = DigiOccupancy_Minus_.at(ilayer)->getBinContent(i)/nevent;
      DigiOccupancy_Minus_.at(ilayer)->setBinContent(i, binValue);
    }

  }//loop over layers
  
  name.str(""); name << "SUMOfDigiOccupancy_Plus";
  for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
    if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
      MeanDigiOccupancy_Plus_.push_back(hgcalMEs[ih]);
    }
  }

  name.str(""); name << "SUMOfDigiOccupancy_Plus";
  for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
    if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
      MeanDigiOccupancy_Minus_.push_back(hgcalMEs[ih]);
    }
  }
  return 1;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HGCalDigiClient);
