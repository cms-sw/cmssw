#include "HcalRealisticZS.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

using namespace std;

#include <iostream>

HcalRealisticZS::HcalRealisticZS(edm::ParameterSet const& conf):
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel"))
{
  bool markAndPass=conf.getParameter<bool>("markAndPass");


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

      algo_=std::auto_ptr<HcalZSAlgoRealistic>
	(new HcalZSAlgoRealistic (markAndPass,
				  conf.getParameter<int>("HBlevel"),
				  conf.getParameter<int>("HElevel"),
				  conf.getParameter<int>("HOlevel"),
				  conf.getParameter<int>("HFlevel"), 
				  HBsearchTS,
				  HEsearchTS,
				  HOsearchTS,
				  HFsearchTS
				  ));
      
    }
    else {

      algo_=std::auto_ptr<HcalZSAlgoRealistic>
	(new HcalZSAlgoRealistic(markAndPass,				  
				 HBsearchTS,
				 HEsearchTS,
				 HOsearchTS,
				 HFsearchTS));    
    }

    produces<HBHEDigiCollection>();
    produces<HODigiCollection>();
    produces<HFDigiCollection>();
    
}
    
HcalRealisticZS::~HcalRealisticZS() {
  algo_->clearDbService();
}
    
void HcalRealisticZS::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
 
  edm::Handle<HBHEDigiCollection> hbhe;    
  edm::Handle<HODigiCollection> ho;    
  edm::Handle<HFDigiCollection> hf;    

  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  algo_->setDbService(conditions.product());

  e.getByLabel(inputLabel_,hbhe);
  
  // create empty output
  std::auto_ptr<HBHEDigiCollection> zs_hbhe(new HBHEDigiCollection);
  
  e.getByLabel(inputLabel_,ho);
  
  // create empty output
  std::auto_ptr<HODigiCollection> zs_ho(new HODigiCollection);
  
  e.getByLabel(inputLabel_,hf);
  
  // create empty output
  std::auto_ptr<HFDigiCollection> zs_hf(new HFDigiCollection);
  
  //run the algorithm

  algo_->suppress(*(hbhe.product()),*zs_hbhe);
  algo_->suppress(*(ho.product()),*zs_ho);
  algo_->suppress(*(hf.product()),*zs_hf);

  
  edm::LogInfo("HcalZeroSuppression") << "Suppression (HBHE) input " << hbhe->size() << " digis, output " << zs_hbhe->size() << " digis" 
				      <<  " (HO) input " << ho->size() << " digis, output " << zs_ho->size() << " digis"
				      <<  " (HF) input " << hf->size() << " digis, output " << zs_hf->size() << " digis";
  

    // return result
    e.put(zs_hbhe);
    e.put(zs_ho);
    e.put(zs_hf);

}
