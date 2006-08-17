using namespace std;
#include "SimCalorimetry/HcalZeroSuppressionProducers/interface/HcalSimpleAmplitudeZS.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

HcalSimpleAmplitudeZS::HcalSimpleAmplitudeZS(edm::ParameterSet const& conf):
  algo_(conf.getParameter<int>("level"),conf.getParameter<int>("firstSample"),conf.getParameter<int>("samplesToAdd"),conf.getParameter<bool>("twoSided")),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel"))
{
  std::string subd=conf.getParameter<std::string>("Subdetector");
  if (!strcasecmp(subd.c_str(),"HBHE")) {
    subdet_=HcalBarrel;
    produces<HBHEDigiCollection>();
  } else if (!strcasecmp(subd.c_str(),"HO")) {
    subdet_=HcalOuter;
    produces<HODigiCollection>();
  } else if (!strcasecmp(subd.c_str(),"HF")) {
    subdet_=HcalForward;
    produces<HFDigiCollection>();
  } else {
    throw cms::Exception("Configuration") << "HcalSimpleAmplitudeZS is not associated with a specific subdetector!";
  }       
  
}
    
HcalSimpleAmplitudeZS::~HcalSimpleAmplitudeZS() {
}
    
void HcalSimpleAmplitudeZS::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  
  if (subdet_==HcalBarrel || subdet_==HcalEndcap) {
    edm::Handle<HBHEDigiCollection> digi;
    
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HBHEDigiCollection> zs(new HBHEDigiCollection);
    // run the algorithm
    algo_.suppress(*conditions,*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HBHE) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs);
  } else if (subdet_==HcalOuter) {
    edm::Handle<HODigiCollection> digi;
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HODigiCollection> zs(new HODigiCollection);
    // run the algorithm
    algo_.suppress(*conditions,*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HO) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs);    
  } else if (subdet_==HcalForward) {
    edm::Handle<HFDigiCollection> digi;
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HFDigiCollection> zs(new HFDigiCollection);
    // run the algorithm
    algo_.suppress(*conditions,*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HF) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs);     
  }
}
