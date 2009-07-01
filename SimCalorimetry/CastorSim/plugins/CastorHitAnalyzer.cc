#include "SimCalorimetry/CastorSim/plugins/CastorHitAnalyzer.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>


CastorHitAnalyzer::CastorHitAnalyzer(edm::ParameterSet const& conf) 
  : hitReadoutName_("CastorHits"),
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
edm::Handle<CrossingFrame<PCaloHit> > castorcf;
e.getByLabel("mix", "CastorFI",castorcf);
  
  // access to SimHits
std::auto_ptr<MixCollection<PCaloHit> > hits(new MixCollection<PCaloHit>(castorcf.product()));
    castorAnalyzer_.fillHits(*hits);
    CastorHitAnalyzerImpl::analyze<CastorRecHitCollection>(e, castorAnalyzer_);
  }



