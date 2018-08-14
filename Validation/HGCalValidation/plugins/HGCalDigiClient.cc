#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"


class HGCalDigiClient : public DQMEDHarvester {
 
private:
  std::string nameDetector_;
  int         verbosity_;
  int         layers_;

public:
  explicit HGCalDigiClient(const edm::ParameterSet&);
  ~HGCalDigiClient() override { }

  void beginRun(const edm::Run& run, const edm::EventSetup& c) override;
  void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) override;
  void runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig);   
  int  digisEndjob(const std::vector<MonitorElement*> &hcalMEs);
};

HGCalDigiClient::HGCalDigiClient(const edm::ParameterSet& iConfig) :
  nameDetector_(iConfig.getParameter<std::string>("DetectorName")),
  verbosity_(iConfig.getUntrackedParameter<int>("Verbosity",0)) { }

void HGCalDigiClient::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) { 
  
  if (nameDetector_ == "HCal") {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
    const HcalDDDRecConstants *hcons  = &(*pHRNDC);
    layers_ = hcons->getMaxDepth(1);
  } else {
    edm::ESHandle<HGCalDDDConstants>  pHGDC;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, pHGDC);
    const HGCalDDDConstants & hgcons_ = (*pHGDC);
    layers_ = hgcons_.layers(true);
  }
}

void HGCalDigiClient::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) {
  runClient_(ib, ig);
}

void HGCalDigiClient::runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig) {

  ig.setCurrentFolder("/"); 
  if (verbosity_) edm::LogVerbatim("HGCalValidation") << "\nrunClient";
  std::vector<MonitorElement*> hgcalMEs;
  std::vector<std::string> fullDirPath = ig.getSubdirs();

  for (unsigned int i=0; i<fullDirPath.size(); i++) {
    if (verbosity_) 
      edm::LogVerbatim("HGCalValidation") << "\nfullPath: " << fullDirPath.at(i);
    ig.setCurrentFolder(fullDirPath.at(i));
    std::vector<std::string> fullSubDirPath = ig.getSubdirs();

    for (unsigned int j=0; j<fullSubDirPath.size(); j++) {
      if (verbosity_) 
	edm::LogVerbatim("HGCalValidation") << "fullSubPath: " << fullSubDirPath.at(j);
      std::string nameDirectory = "HGCAL/HGCalDigisV/"+nameDetector_;
      if (strcmp(fullSubDirPath.at(j).c_str(), nameDirectory.c_str()) == 0) {
        hgcalMEs = ig.getContents(fullSubDirPath.at(j));
        if (verbosity_) 
	  edm::LogVerbatim("HGCalValidation") << "hgcalMES size : " << hgcalMEs.size();
        if (!digisEndjob(hgcalMEs)) 
	  edm::LogVerbatim("HGCalValidation") << "\nError in DigisEndjob!";
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
  int    nbinsx, nbinsy;
  
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
    if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
      MeanDigiOccupancy_Plus_.push_back(hgcalMEs[ih]);
      unsigned indx = MeanDigiOccupancy_Plus_.size()-1;
      for(int ilayer=0; ilayer < (int)layers_; ++ilayer) {
	double meanVal = DigiOccupancy_Plus_.at(ilayer)->getMean();
	MeanDigiOccupancy_Plus_[indx]->setBinContent(ilayer+1, meanVal);
      }
      break;
    }
  }

  name.str(""); name << "SUMOfDigiOccupancy_Plus";
  for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
    if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
      MeanDigiOccupancy_Minus_.push_back(hgcalMEs[ih]);
      unsigned indx = MeanDigiOccupancy_Minus_.size()-1;
      for(int ilayer=0; ilayer < (int)layers_; ++ilayer) {
	double meanVal = DigiOccupancy_Minus_.at(ilayer)->getMean();
	MeanDigiOccupancy_Minus_[indx]->setBinContent(ilayer+1, meanVal);
      }
      break;
    }
  }
  return 1;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HGCalDigiClient);
