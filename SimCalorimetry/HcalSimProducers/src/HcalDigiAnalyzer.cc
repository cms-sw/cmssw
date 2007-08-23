#include "SimCalorimetry/HcalSimProducers/src/HcalDigiAnalyzer.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>


HcalDigiAnalyzer::HcalDigiAnalyzer(edm::ParameterSet const& conf) 
: hitReadoutName_("HcalHits"),
  simParameterMap_(),
  hbheFilter_(),
  hoFilter_(),
  hfFilter_(),
  hbheHitAnalyzer_("HBHEDigi", 1., &simParameterMap_, &hbheFilter_),
  hoHitAnalyzer_("HODigi", 1., &simParameterMap_, &hoFilter_),
  hfHitAnalyzer_("HFDigi", 1., &simParameterMap_, &hfFilter_),
  zdcHitAnalyzer_("ZDCDigi", 1., &simParameterMap_, &zdcFilter_),
  hbheDigiStatistics_("HBHEDigi", 4, 10., 6., 0.1, 0.5, hbheHitAnalyzer_),
  hoDigiStatistics_("HODigi", 4, 10., 6., 0.1, 0.5, hoHitAnalyzer_),
  hfDigiStatistics_("HFDigi", 3, 10., 6., 0.1, 0.5, hfHitAnalyzer_),
  zdcDigiStatistics_("ZDCDigi", 3, 10., 6., 0.1, 0.5, zdcHitAnalyzer_)
{
}


namespace HcalDigiAnalyzerImpl {
  template<class Collection>
  void analyze(edm::Event const& e, HcalDigiStatistics & statistics) {
    try {
      edm::Handle<Collection> digis;
      e.getByType(digis);
      for(unsigned i = 0; i < digis->size(); ++i) {
        statistics.analyze((*digis)[i]);
      }
    }
    catch (...) {
      edm::LogError("HcalDigiAnalyzer") << "Could not find Hcal Digi Container ";
    }
  }
}


void HcalDigiAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& c) {
  edm::Handle<edm::PCaloHitContainer> hits;
  try{
    hitReadoutName_ = "HcalHits";
    e.getByLabel("g4SimHits",hitReadoutName_, hits);
    
  } catch(...){;}
  try{
    hitReadoutName_ = "ZDCHITS";
    e.getByLabel("g4SimHits",hitReadoutName_, hits);
    
  } catch(...){;}
  hbheHitAnalyzer_.fillHits(*hits);
  hoHitAnalyzer_.fillHits(*hits);
  hfHitAnalyzer_.fillHits(*hits);
  zdcHitAnalyzer_.fillHits(*hits);
  HcalDigiAnalyzerImpl::analyze<HBHEDigiCollection>(e, hbheDigiStatistics_);
  HcalDigiAnalyzerImpl::analyze<HODigiCollection>(e, hoDigiStatistics_);
  HcalDigiAnalyzerImpl::analyze<HFDigiCollection>(e, hfDigiStatistics_);
  HcalDigiAnalyzerImpl::analyze<ZDCDigiCollection>(e, zdcDigiStatistics_);
}


