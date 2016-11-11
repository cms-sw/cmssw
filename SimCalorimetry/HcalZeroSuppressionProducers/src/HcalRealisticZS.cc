#include "HcalRealisticZS.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <iostream>

HcalRealisticZS::HcalRealisticZS(edm::ParameterSet const& conf):
  inputLabel_(conf.getParameter<std::string>("digiLabel")) {

  bool markAndPass=conf.getParameter<bool>("markAndPass");

  // register for data access
  tok_hbhe_ = consumes<HBHEDigiCollection>(edm::InputTag(inputLabel_));
  tok_ho_ = consumes<HODigiCollection>(edm::InputTag(inputLabel_));
  tok_hf_ = consumes<HFDigiCollection>(edm::InputTag(inputLabel_));
  tok_hfQIE10_ = consumes<QIE10DigiCollection>(edm::InputTag(inputLabel_, "HFQIE10DigiCollection"));
  tok_hbheQIE11_ = consumes<QIE11DigiCollection>(edm::InputTag(inputLabel_, "HBHEQIE11DigiCollection"));


  std::vector<int> tmp = conf.getParameter<std::vector<int> >("HBregion");

  if(tmp[0]<0 || tmp[0]>9 || tmp[1]<0 || tmp[1]>9 || tmp[0]>tmp[1]) {
    edm::LogError("HcalZeroSuppression") << "ZS(HB) region error: " 
					 << tmp[0] << ":" <<tmp[1];
    tmp[0]=0; tmp[1]=9;
  }

  std::pair<int,int> HBsearchTS (tmp[0],tmp[1]);
  
  tmp = conf.getParameter<std::vector<int> >("HEregion");
  if(tmp[0]<0 || tmp[0]>9 || tmp[1]<0 || tmp[1]>9 || tmp[0]>tmp[1]) {
    edm::LogError("HcalZeroSuppression") << "ZS(HE) region error: " 
					 << tmp[0] << ":" <<tmp[1];
    tmp[0]=0; tmp[1]=9;
  }
  std::pair<int,int> HEsearchTS (tmp[0],tmp[1]);
  
  tmp = conf.getParameter<std::vector<int> >("HOregion");
  if(tmp[0]<0 || tmp[0]>9 || tmp[1]<0 || tmp[1]>9 || tmp[0]>tmp[1]) {
    edm::LogError("HcalZeroSuppression") << "ZS(HO) region error: " 
					 << tmp[0] << ":" <<tmp[1];
    tmp[0]=0; tmp[1]=9;
  }
  std::pair<int,int> HOsearchTS (tmp[0],tmp[1]);
  
  tmp = conf.getParameter<std::vector<int> >("HFregion");
  if(tmp[0]<0 || tmp[0]>9 || tmp[1]<0 || tmp[1]>9 || tmp[0]>tmp[1]) {
    edm::LogError("HcalZeroSuppression") << "ZS(HF) region error: " 
					 << tmp[0] << ":" <<tmp[1];
    tmp[0]=0; tmp[1]=9;
  }
  std::pair<int,int> HFsearchTS (tmp[0],tmp[1]);
      
  
    //this constructor will be called if useConfigZSvalues is set to 1 in
    //HcalZeroSuppressionProducers/python/hcalDigisRealistic_cfi.py
    //which means that channel-by-channel ZS thresholds from DB will NOT be used
    if ( conf.getParameter<int>("useConfigZSvalues") ) {

      algo_.reset(
	 new HcalZSAlgoRealistic (markAndPass,
				  conf.getParameter<int>("HBlevel"),
				  conf.getParameter<int>("HElevel"),
				  conf.getParameter<int>("HOlevel"),
				  conf.getParameter<int>("HFlevel"), 
				  HBsearchTS,
				  HEsearchTS,
				  HOsearchTS,
				  HFsearchTS
				  ));
      
    } else {

      algo_.reset(
	 new HcalZSAlgoRealistic(markAndPass,				  
				 HBsearchTS,
				 HEsearchTS,
				 HOsearchTS,
				 HFsearchTS));    
    }

    produces<HBHEDigiCollection>();
    produces<HODigiCollection>();
    produces<HFDigiCollection>();
    produces<QIE10DigiCollection>("HFQIE10DigiCollection");
    produces<QIE11DigiCollection>("HBHEQIE11DigiCollection");
   
}
    
HcalRealisticZS::~HcalRealisticZS() {
  algo_->clearDbService();
}
    
void HcalRealisticZS::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
 
  edm::Handle<HBHEDigiCollection> hbhe;    
  edm::Handle<HODigiCollection> ho;    
  edm::Handle<HFDigiCollection> hf;
  edm::Handle<QIE10DigiCollection> hfQIE10;
  edm::Handle<QIE11DigiCollection> hbheQIE11;

  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  algo_->setDbService(conditions.product());

  e.getByToken(tok_hbhe_,hbhe);
  
  // create empty output
  std::unique_ptr<HBHEDigiCollection> zs_hbhe(new HBHEDigiCollection);
  
  e.getByToken(tok_ho_,ho);
  
  // create empty output
  std::unique_ptr<HODigiCollection> zs_ho(new HODigiCollection);
  
  e.getByToken(tok_hf_,hf);
  
  // create empty output
  std::unique_ptr<HFDigiCollection> zs_hf(new HFDigiCollection);
  
  e.getByToken(tok_hfQIE10_,hfQIE10);
  e.getByToken(tok_hbheQIE11_,hbheQIE11);
  
  // create empty output
  std::unique_ptr<QIE10DigiCollection> zs_hfQIE10(new QIE10DigiCollection);
  std::unique_ptr<QIE11DigiCollection> zs_hbheQIE11(new QIE11DigiCollection);
  
  //run the algorithm

  algo_->suppress(*(hbhe.product()),*zs_hbhe);
  algo_->suppress(*(ho.product()),*zs_ho);
  algo_->suppress(*(hf.product()),*zs_hf);
  algo_->suppress(*(hfQIE10.product()),*zs_hfQIE10);
  algo_->suppress(*(hbheQIE11.product()),*zs_hbheQIE11);

  
  edm::LogInfo("HcalZeroSuppression") << "Suppression (HBHE) input " << hbhe->size() << " digis, output " << zs_hbhe->size() << " digis" 
				      <<  " (HO) input " << ho->size() << " digis, output " << zs_ho->size() << " digis"
				      <<  " (HF) input " << hf->size() << " digis, output " << zs_hf->size() << " digis"
				      <<  " (HFQIE10) input " << hfQIE10->size() << " digis, output " << zs_hfQIE10->size() << " digis"
				      <<  " (HBHEQIE11) input " << hbheQIE11->size() << " digis, output " << zs_hbheQIE11->size() << " digis";
  

    // return result
    e.put(std::move(zs_hbhe));
    e.put(std::move(zs_ho));
    e.put(std::move(zs_hf));
    e.put(std::move(zs_hfQIE10),"HFQIE10DigiCollection");
    e.put(std::move(zs_hbheQIE11),"HBHEQIE11DigiCollection");

}
