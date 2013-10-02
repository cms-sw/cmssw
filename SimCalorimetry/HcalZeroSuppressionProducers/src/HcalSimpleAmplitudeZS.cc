#include "HcalSimpleAmplitudeZS.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
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
  inputLabel_(conf.getParameter<std::string>("digiLabel"))
{

  // register for data access
  tok_hbhe_ = consumes<HBHEDigiCollection>(edm::InputTag(inputLabel_));
  tok_ho_ = consumes<HODigiCollection>(edm::InputTag(inputLabel_));
  tok_hf_ = consumes<HFDigiCollection>(edm::InputTag(inputLabel_));
    tok_hbheUpgrade_ = consumes<HBHEUpgradeDigiCollection>(edm::InputTag(inputLabel_, "HBHEUpgradeDigiCollection"));
    tok_hfUpgrade_ = consumes<HFUpgradeDigiCollection>(edm::InputTag(inputLabel_, "HFUpgradeDigiCollection"));

  const edm::ParameterSet& psHBHE=conf.getParameter<edm::ParameterSet>("hbhe");
  bool markAndPass=psHBHE.getParameter<bool>("markAndPass");
  hbhe_=std::auto_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,
							     psHBHE.getParameter<int>("level"),
							     psHBHE.getParameter<int>("firstSample"),
							     psHBHE.getParameter<int>("samplesToAdd"),
							     psHBHE.getParameter<bool>("twoSided")));
  produces<HBHEDigiCollection>();  
  hbheUpgrade_=std::auto_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,
								    psHBHE.getParameter<int>("level"),
								    psHBHE.getParameter<int>("firstSample"),
								    psHBHE.getParameter<int>("samplesToAdd"),
								    psHBHE.getParameter<bool>("twoSided")));
  produces<HBHEUpgradeDigiCollection>("HBHEUpgradeDigiCollection");  

  const edm::ParameterSet& psHO=conf.getParameter<edm::ParameterSet>("ho");
  markAndPass=psHO.getParameter<bool>("markAndPass");
  ho_=std::auto_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,
							   psHO.getParameter<int>("level"),
							   psHO.getParameter<int>("firstSample"),
							   psHO.getParameter<int>("samplesToAdd"),
							   psHO.getParameter<bool>("twoSided")));
  produces<HODigiCollection>();
  
  const edm::ParameterSet& psHF=conf.getParameter<edm::ParameterSet>("hf");
  markAndPass=psHF.getParameter<bool>("markAndPass");
  hf_=std::auto_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,	
							   psHF.getParameter<int>("level"),
							   psHF.getParameter<int>("firstSample"),
							   psHF.getParameter<int>("samplesToAdd"),
							   psHF.getParameter<bool>("twoSided")));
  produces<HFDigiCollection>();
  hfUpgrade_=std::auto_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,	
								  psHF.getParameter<int>("level"),
								  psHF.getParameter<int>("firstSample"),
								  psHF.getParameter<int>("samplesToAdd"),
								  psHF.getParameter<bool>("twoSided")));
  produces<HFUpgradeDigiCollection>("HFUpgradeDigiCollection");  
  
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
    e.getByToken(tok_hbhe_,digi);
    
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
    e.getByToken(tok_ho_,digi);
    
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
    e.getByToken(tok_hf_,digi);
    
    // create empty output
    std::auto_ptr<HFDigiCollection> zs(new HFDigiCollection);
    // run the algorithm
    hf_->suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HF) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs);     
    hf_->done();
  }
  {
    hbheUpgrade_->prepare(&(*conditions));
    edm::Handle<HBHEUpgradeDigiCollection> digi;    
    e.getByToken(tok_hbheUpgrade_,digi);
    
    // create empty output
    std::auto_ptr<HBHEUpgradeDigiCollection> zs(new HBHEUpgradeDigiCollection);
    // run the algorithm
    hbheUpgrade_->suppress(*(digi.product()),*zs);
    
    edm::LogInfo("HcalZeroSuppression") << "Suppression (HBHEUpgrade) input " << digi->size() << " digis, output " << zs->size() << " digis";
    
    // return result
    e.put(zs,"HBHEUpgradeDigiCollection");
    hbheUpgrade_->done();
  } 
  {
    hfUpgrade_->prepare(&(*conditions));
    edm::Handle<HFUpgradeDigiCollection> digi;
    e.getByToken(tok_hfUpgrade_,digi);
    
    // create empty output
    std::auto_ptr<HFUpgradeDigiCollection> zs(new HFUpgradeDigiCollection);
    // run the algorithm
    hfUpgrade_->suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HFUpgrade) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(zs, "HFUpgradeDigiCollection");     
    hfUpgrade_->done();
  }

}
