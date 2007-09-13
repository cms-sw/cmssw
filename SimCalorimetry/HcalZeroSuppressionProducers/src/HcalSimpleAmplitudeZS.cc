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
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel"))
{
  const edm::ParameterSet& psHBHE=conf.getParameter<edm::ParameterSet>("hbhe");
  hbhe_=std::auto_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(
							     (psHBHE.getParameter<bool>("triggerOR")?(HcalZeroSuppressionAlgo::zs_TriggerTowerOR):(HcalZeroSuppressionAlgo::zs_SingleChannel)),
							     psHBHE.getParameter<int>("level"),
							     psHBHE.getParameter<int>("firstSample"),
							     psHBHE.getParameter<int>("samplesToAdd"),
							     psHBHE.getParameter<bool>("twoSided")));
  produces<HBHEDigiCollection>();
  
  const edm::ParameterSet& psHO=conf.getParameter<edm::ParameterSet>("ho");
  ho_=std::auto_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(
							     (psHO.getParameter<bool>("triggerOR")?(HcalZeroSuppressionAlgo::zs_TriggerTowerOR):(HcalZeroSuppressionAlgo::zs_SingleChannel)),
							     psHO.getParameter<int>("level"),
							     psHO.getParameter<int>("firstSample"),
							     psHO.getParameter<int>("samplesToAdd"),
							     psHO.getParameter<bool>("twoSided")));
  produces<HODigiCollection>();
  
  const edm::ParameterSet& psHF=conf.getParameter<edm::ParameterSet>("hf");
  hf_=std::auto_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(
							     (psHF.getParameter<bool>("triggerOR")?(HcalZeroSuppressionAlgo::zs_TriggerTowerOR):(HcalZeroSuppressionAlgo::zs_SingleChannel)),
							     psHF.getParameter<int>("level"),
							     psHF.getParameter<int>("firstSample"),
							     psHF.getParameter<int>("samplesToAdd"),
							     psHF.getParameter<bool>("twoSided")));
  produces<HFDigiCollection>();
  
}
    
HcalSimpleAmplitudeZS::~HcalSimpleAmplitudeZS() {
}
    
void HcalSimpleAmplitudeZS::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  

  {
    hbhe_->prepare(&(*conditions));
    edm::Handle<HBHEDigiCollection> digi;    
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HBHEDigiCollection> zs(new HBHEDigiCollection);
    // run the algorithm
    hbhe_->suppress(*(digi.product()),*zs);
    
    edm::LogInfo("HcalZeroSuppression") << "Suppression (HBHE) input " << digi->size() << " digis, output " << zs->size() << " digis";
    
    // return result
    e.put(zs);
    hbhe_->done();
  } 
  {
    ho_->prepare(&(*conditions));
    edm::Handle<HODigiCollection> digi;
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HODigiCollection> zs(new HODigiCollection);
    // run the algorithm
    ho_->suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HO) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs);    
    ho_->done();
  } 
  {
    hf_->prepare(&(*conditions));
    edm::Handle<HFDigiCollection> digi;
    e.getByLabel(inputLabel_,digi);
    
    // create empty output
    std::auto_ptr<HFDigiCollection> zs(new HFDigiCollection);
    // run the algorithm
    hf_->suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HF) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs);     
    hf_->done();
  }

}
