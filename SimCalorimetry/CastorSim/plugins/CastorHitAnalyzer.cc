#include "SimCalorimetry/CastorSim/plugins/CastorHitAnalyzer.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>


CastorHitAnalyzer::CastorHitAnalyzer(edm::ParameterSet const& conf) 
  : hitReadoutName_("CASTORHITS"),
  simParameterMap_(),
  castorFilter_(),
  castorAnalyzer_("CASTOR", 1., &simParameterMap_, &castorFilter_)
{
}


namespace CastorHitAnalyzerImpl {
  template<class Collection>
  void analyze(edm::Event const& e, CaloHitAnalyzer & analyzer) {
    edm::Handle<Collection> recHits;
    e.getByType(recHits);
    if (!recHits.isValid()) {
      edm::LogError("CastorHitAnalyzer") << "Could not find Castor RecHitContainer ";
    } else {
      for(unsigned i = 0 ; i < recHits->size(); ++i) {
        analyzer.analyze((*recHits)[i].id().rawId(), (*recHits)[i].energy());
      }
    }
  }
}


void CastorHitAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& c) {
  edm::Handle<edm::PCaloHitContainer> hits;
  hitReadoutName_ = "CASTORHITS";
  e.getByLabel("g4SimHits",hitReadoutName_, hits);
  if (hits.isValid()) {
    castorAnalyzer_.fillHits(*hits);
    CastorHitAnalyzerImpl::analyze<CastorRecHitCollection>(e, castorAnalyzer_);
  }
}


