#include "SimCalorimetry/HcalSimProducers/src/HcalHitAnalyzer.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>


HcalHitAnalyzer::HcalHitAnalyzer(edm::ParameterSet const& conf) 
  : hitReadoutName_("HcalHits"),
  simParameterMap_(),
  hbheFilter_(),
  hoFilter_(),
  hfFilter_(),
  zdcFilter_(),
  hbheAnalyzer_("HBHE", 1., &simParameterMap_, &hbheFilter_),
  hoAnalyzer_("HO", 1., &simParameterMap_, &hoFilter_),
  hfAnalyzer_("HF", 1., &simParameterMap_, &hfFilter_),
  zdcAnalyzer_("ZDC", 1., &simParameterMap_, &zdcFilter_)
{
}


namespace HcalHitAnalyzerImpl {
  template<class Collection>
  void analyze(edm::Event const& e, CaloHitAnalyzer & analyzer) {
    try {
      edm::Handle<Collection> recHits;
      e.getByType(recHits);
      for(unsigned i = 0 ; i < recHits->size(); ++i) {
        analyzer.analyze((*recHits)[i].id().rawId(), (*recHits)[i].energy());
      }
    }
    catch (...) {
      edm::LogError("HcalHitAnalyzer") << "Could not find Hcal RecHitContainer ";
    }
  }
}


void HcalHitAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& c) {
  edm::Handle<edm::PCaloHitContainer> hits;
  try{
    hitReadoutName_ = "HcalHits";
    e.getByLabel("g4SimHits",hitReadoutName_, hits);
    
  } catch(...){;}
  try{
    hitReadoutName_ = "ZDCHITS";
    e.getByLabel("g4SimHits",hitReadoutName_, hits);
    
  } catch(...){;}
  
  hbheAnalyzer_.fillHits(*hits);
  hoAnalyzer_.fillHits(*hits);
  hfAnalyzer_.fillHits(*hits);
  zdcAnalyzer_.fillHits(*hits);
  HcalHitAnalyzerImpl::analyze<HBHERecHitCollection>(e, hbheAnalyzer_);
  HcalHitAnalyzerImpl::analyze<HORecHitCollection>(e, hoAnalyzer_);
  HcalHitAnalyzerImpl::analyze<HFRecHitCollection>(e, hfAnalyzer_);
  HcalHitAnalyzerImpl::analyze<ZDCRecHitCollection>(e, zdcAnalyzer_);
}


