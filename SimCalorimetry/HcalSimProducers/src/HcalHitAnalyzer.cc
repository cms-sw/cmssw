#include "SimCalorimetry/HcalSimProducers/src/HcalHitAnalyzer.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>


HcalHitAnalyzer::HcalHitAnalyzer(edm::ParameterSet const& conf) 
: simParameterMap_(conf),
  hbheFilter_(),
  hoFilter_(),
  hfFilter_(true),
  zdcFilter_(),
  hbheAnalyzer_("HBHE", 1., &simParameterMap_, &hbheFilter_),
  hoAnalyzer_("HO", 1., &simParameterMap_, &hoFilter_),
  hfAnalyzer_("HF", 1., &simParameterMap_, &hfFilter_),
  zdcAnalyzer_("ZDC", 1., &simParameterMap_, &zdcFilter_),
  hbheRecHitCollectionTag_(conf.getParameter<edm::InputTag>("hbheRecHitCollectionTag")),
  hoRecHitCollectionTag_(conf.getParameter<edm::InputTag>("hoRecHitCollectionTag")),
  hfRecHitCollectionTag_(conf.getParameter<edm::InputTag>("hfRecHitCollectionTag"))
{
}


namespace HcalHitAnalyzerImpl {
  template<class Collection>
  void analyze(edm::Event const& e, CaloHitAnalyzer & analyzer, edm::InputTag& tag) {
    edm::Handle<Collection> recHits;
    e.getByLabel(tag, recHits);
    for(unsigned i = 0 ; i < recHits->size(); ++i) {
      analyzer.analyze((*recHits)[i].id().rawId(), (*recHits)[i].energy());
    }
  }
}


void HcalHitAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& c) {
   // Step A: Get Inputs
  edm::Handle<CrossingFrame<PCaloHit> > cf, zdccf;
  e.getByLabel("mix", "g4SimHitsHcalHits",cf);
  //e.getByLabel("mix", "ZDCHits", zdccf);

  // test access to SimHits for HcalHits and ZDC hits
  std::auto_ptr<MixCollection<PCaloHit> > hits(new MixCollection<PCaloHit>(cf.product()));
  //std::auto_ptr<MixCollection<PCaloHit> > zdcHits(new MixCollection<PCaloHit>(zdccf.product()));
  hbheAnalyzer_.fillHits(*hits); 
  //hoAnalyzer_.fillHits(*hits);
  //hfAnalyzer_.fillHits(*hits);
  //zdcAnalyzer_.fillHits(*hits);
  HcalHitAnalyzerImpl::analyze<HBHERecHitCollection>(e, hbheAnalyzer_, hbheRecHitCollectionTag_);
  HcalHitAnalyzerImpl::analyze<HORecHitCollection>(e, hoAnalyzer_, hoRecHitCollectionTag_);
  HcalHitAnalyzerImpl::analyze<HFRecHitCollection>(e, hfAnalyzer_, hfRecHitCollectionTag_);
  //HcalHitAnalyzerImpl::analyze<ZDCRecHitCollection>(e, zdcAnalyzer_);
}
