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
  tok_hfQIE10_ = consumes<QIE10DigiCollection>(edm::InputTag(inputLabel_, "HFQIE10DigiCollection"));
  tok_hbheQIE11_ = consumes<QIE11DigiCollection>(edm::InputTag(inputLabel_, "HBHEQIE11DigiCollection"));

  const edm::ParameterSet& psHBHE=conf.getParameter<edm::ParameterSet>("hbhe");
  bool markAndPass=psHBHE.getParameter<bool>("markAndPass");
  hbhe_=std::unique_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,
							     psHBHE.getParameter<int>("level"),
							     psHBHE.getParameter<int>("firstSample"),
							     psHBHE.getParameter<int>("samplesToAdd"),
							     psHBHE.getParameter<bool>("twoSided")));
  produces<HBHEDigiCollection>();  
  hbheQIE11_=std::unique_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,	
								  psHBHE.getParameter<int>("level"),
								  psHBHE.getParameter<int>("firstSample"),
								  psHBHE.getParameter<int>("samplesToAdd"),
								  psHBHE.getParameter<bool>("twoSided")));
  produces<QIE11DigiCollection>("HBHEQIE11DigiCollection");  
  
  const edm::ParameterSet& psHO=conf.getParameter<edm::ParameterSet>("ho");
  markAndPass=psHO.getParameter<bool>("markAndPass");
  ho_=std::unique_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,
							   psHO.getParameter<int>("level"),
							   psHO.getParameter<int>("firstSample"),
							   psHO.getParameter<int>("samplesToAdd"),
							   psHO.getParameter<bool>("twoSided")));
  produces<HODigiCollection>();
  
  const edm::ParameterSet& psHF=conf.getParameter<edm::ParameterSet>("hf");
  markAndPass=psHF.getParameter<bool>("markAndPass");
  hf_=std::unique_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,
							   psHF.getParameter<int>("level"),
							   psHF.getParameter<int>("firstSample"),
							   psHF.getParameter<int>("samplesToAdd"),
							   psHF.getParameter<bool>("twoSided")));
  produces<HFDigiCollection>();
  hfQIE10_=std::unique_ptr<HcalZSAlgoEnergy>(new HcalZSAlgoEnergy(markAndPass,
								  psHF.getParameter<int>("level"),
								  psHF.getParameter<int>("firstSample"),
								  psHF.getParameter<int>("samplesToAdd"),
								  psHF.getParameter<bool>("twoSided")));
  produces<QIE10DigiCollection>("HFQIE10DigiCollection");  
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
    std::unique_ptr<HBHEDigiCollection> zs(new HBHEDigiCollection);
    // run the algorithm
    hbhe_->suppress(*(digi.product()),*zs);
    
    edm::LogInfo("HcalZeroSuppression") << "Suppression (HBHE) input " << digi->size() << " digis, output " << zs->size() << " digis";
    
    // return result
    e.put(std::move(zs));
    hbhe_->done();
  } 
  {
    ho_->prepare(&(*conditions));
    edm::Handle<HODigiCollection> digi;
    e.getByToken(tok_ho_,digi);
    
    // create empty output
    std::unique_ptr<HODigiCollection> zs(new HODigiCollection);
    // run the algorithm
    ho_->suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HO) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(std::move(zs));
    ho_->done();
  } 
  {
    hf_->prepare(&(*conditions));
    edm::Handle<HFDigiCollection> digi;
    e.getByToken(tok_hf_,digi);
    
    // create empty output
    std::unique_ptr<HFDigiCollection> zs(new HFDigiCollection);
    // run the algorithm
    hf_->suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HF) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(std::move(zs));
    hf_->done();
  }
  {
    hfQIE10_->prepare(&(*conditions));
    edm::Handle<QIE10DigiCollection> digi;
    e.getByToken(tok_hfQIE10_,digi);
    
    // create empty output
    std::unique_ptr<QIE10DigiCollection> zs(new QIE10DigiCollection);
    // run the algorithm
    hfQIE10_->suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HFQIE10) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(std::move(zs), "HFQIE10DigiCollection");
    hfQIE10_->done();
  }
  {
    hbheQIE11_->prepare(&(*conditions));
    edm::Handle<QIE11DigiCollection> digi;
    e.getByToken(tok_hbheQIE11_,digi);
    
    // create empty output
    std::unique_ptr<QIE11DigiCollection> zs(new QIE11DigiCollection);
    // run the algorithm
    hbheQIE11_->suppress(*(digi.product()),*zs);

    edm::LogInfo("HcalZeroSuppression") << "Suppression (HBHEQIE11) input " << digi->size() << " digis, output " << zs->size() << " digis";

    // return result
    e.put(std::move(zs), "HBHEQIE11DigiCollection");     
    hbheQIE11_->done();
  }
}
