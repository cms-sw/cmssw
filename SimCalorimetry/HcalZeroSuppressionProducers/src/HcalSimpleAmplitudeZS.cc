#include "SimCalorimetry/HcalZeroSuppressionProducers/src/HcalSimpleAmplitudeZS.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
using namespace std;

#include <iostream>

HcalSimpleAmplitudeZS::HcalSimpleAmplitudeZS(edm::ParameterSet const& conf):
  algo_(((conf.getParameter<bool>("triggerOR"))?(HcalZeroSuppressionAlgo::zs_TriggerTowerOR):(HcalZeroSuppressionAlgo::zs_SingleChannel)),
	conf.getParameter<int>("level"),
	conf.getParameter<int>("firstSample"),
	conf.getParameter<int>("samplesToAdd"),
	conf.getParameter<bool>("twoSided")),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel"))
{
  std::string subd=conf.getParameter<std::string>("Subdetector");
  if (!strcasecmp(subd.c_str(),"ALL")) {
    subdets_.insert(HcalBarrel);
    subdets_.insert(HcalOuter);
    subdets_.insert(HcalForward);
    produces<HBHEDigiCollection>();
    produces<HODigiCollection>();
    produces<HFDigiCollection>();
  } else if (!strcasecmp(subd.c_str(),"HBHE")) {
    subdets_.insert(HcalBarrel);
    produces<HBHEDigiCollection>();
  } else if (!strcasecmp(subd.c_str(),"HO")) {
    subdets_.insert(HcalOuter);
    produces<HODigiCollection>();
  } else if (!strcasecmp(subd.c_str(),"HF")) {
    subdets_.insert(HcalForward);
    produces<HFDigiCollection>();
  } else {
    throw cms::Exception("Configuration") << "HcalSimpleAmplitudeZS is not associated with a specific subdetector or with ALL!";
  }       
  
}
    
HcalSimpleAmplitudeZS::~HcalSimpleAmplitudeZS() {
}
    
void HcalSimpleAmplitudeZS::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  

  algo_.prepare(&(*conditions));

  if (subdets_.find(HcalBarrel)!=subdets_.end() ||
      subdets_.find(HcalEndcap)!=subdets_.end()) {
    edm::Handle<HBHEDigiCollection> digi;
    
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HBHEDigiCollection> zs(new HBHEDigiCollection);
    // run the algorithm
    algo_.suppress(*(digi.product()),*zs);
    
    edm::LogInfo("HcalZeroSuppression") << "Suppression (HBHE) input " << digi->size() << " digis, output " << zs->size() << " digis";
    
    // return result
    e.put(zs);
  } 
  if (subdets_.find(HcalOuter)!=subdets_.end()) {
    edm::Handle<HODigiCollection> digi;
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HODigiCollection> zs(new HODigiCollection);
    // run the algorithm
    algo_.suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HO) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs);    
  } 
  if (subdets_.find(HcalForward)!=subdets_.end()) {
    edm::Handle<HFDigiCollection> digi;
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HFDigiCollection> zs(new HFDigiCollection);
    // run the algorithm
    algo_.suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HF) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs);     
  }
  algo_.done();
}
