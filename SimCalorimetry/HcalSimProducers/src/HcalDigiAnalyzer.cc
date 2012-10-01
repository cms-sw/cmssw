#include "SimCalorimetry/HcalSimProducers/src/HcalDigiAnalyzer.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>


HcalDigiAnalyzer::HcalDigiAnalyzer(edm::ParameterSet const& conf) 
: hitReadoutName_("HcalHits"),
  simParameterMap_(),
  hbheFilter_(),
  hoFilter_(),
  hfFilter_(true),
  hbheHitAnalyzer_("HBHEDigi", 1., &simParameterMap_, &hbheFilter_),
  hoHitAnalyzer_("HODigi", 1., &simParameterMap_, &hoFilter_),
  hfHitAnalyzer_("HFDigi", 1., &simParameterMap_, &hfFilter_),
  zdcHitAnalyzer_("ZDCDigi", 1., &simParameterMap_, &zdcFilter_),
  hbheDigiStatistics_("HBHEDigi", 4, 10., 6., 0.1, 0.5, hbheHitAnalyzer_),
  hoDigiStatistics_("HODigi", 4, 10., 6., 0.1, 0.5, hoHitAnalyzer_),
  hfDigiStatistics_("HFDigi", 3, 10., 6., 0.1, 0.5, hfHitAnalyzer_),
  zdcDigiStatistics_("ZDCDigi", 3, 10., 6., 0.1, 0.5, zdcHitAnalyzer_),
  hbheDigiCollectionTag_(conf.getParameter<edm::InputTag>("hbheDigiCollectionTag")),
  hoDigiCollectionTag_(conf.getParameter<edm::InputTag>("hoDigiCollectionTag")),
  hfDigiCollectionTag_(conf.getParameter<edm::InputTag>("hfDigiCollectionTag"))
{
}


namespace HcalDigiAnalyzerImpl {
  template<class Collection>
  void analyze(edm::Event const& e, HcalDigiStatistics & statistics, edm::InputTag& tag) {
    edm::Handle<Collection> digis;
    e.getByLabel(tag, digis);
    for(unsigned i = 0; i < digis->size(); ++i) {
      std::cout << (*digis)[i] << std::endl;
      statistics.analyze((*digis)[i]);
    }
  }
}


void HcalDigiAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& c) {
  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PCaloHit> > cf, zdccf;
  e.getByLabel("mix", "HcalHits",cf);
  //e.getByLabel("mix", "ZDCHits", zdccf);
  
  // test access to SimHits for HcalHits and ZDC hits
  std::auto_ptr<MixCollection<PCaloHit> > hits(new MixCollection<PCaloHit>(cf.product()));
  //std::auto_ptr<MixCollection<PCaloHit> > zdcHits(new MixCollection<PCaloHit>(zdccf.product()));
  hbheHitAnalyzer_.fillHits(*hits);
  hoHitAnalyzer_.fillHits(*hits);
  hfHitAnalyzer_.fillHits(*hits);
  //zdcHitAnalyzer_.fillHits(*zdcHits);
  HcalDigiAnalyzerImpl::analyze<HBHEDigiCollection>(e, hbheDigiStatistics_, hbheDigiCollectionTag_);
  HcalDigiAnalyzerImpl::analyze<HODigiCollection>(e, hoDigiStatistics_, hoDigiCollectionTag_);
  HcalDigiAnalyzerImpl::analyze<HFDigiCollection>(e, hfDigiStatistics_, hfDigiCollectionTag_);
  //HcalDigiAnalyzerImpl::analyze<ZDCDigiCollection>(e, zdcDigiStatistics_);
}
