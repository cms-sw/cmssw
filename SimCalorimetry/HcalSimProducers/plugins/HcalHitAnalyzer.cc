/** Compares HCAL RecHits to SimHit

  \Author Rick Wilkinson, Caltech
*/

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"

#include <iostream>
#include <string>

class HcalHitAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalHitAnalyzer(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

private:
  HcalSimParameterMap simParameterMap_;
  HBHEHitFilter hbheFilter_;
  HOHitFilter hoFilter_;
  HFHitFilter hfFilter_;
  ZDCHitFilter zdcFilter_;
  CaloHitAnalyzer hbheAnalyzer_;
  CaloHitAnalyzer hoAnalyzer_;
  CaloHitAnalyzer hfAnalyzer_;
  CaloHitAnalyzer zdcAnalyzer_;

  edm::InputTag hbheRecHitCollectionTag_;
  edm::InputTag hoRecHitCollectionTag_;
  edm::InputTag hfRecHitCollectionTag_;
};

HcalHitAnalyzer::HcalHitAnalyzer(edm::ParameterSet const &conf)
    : simParameterMap_(conf),
      hbheFilter_(),
      hoFilter_(),
      hfFilter_(),
      zdcFilter_(),
      hbheAnalyzer_("HBHE", 1., &simParameterMap_, &hbheFilter_),
      hoAnalyzer_("HO", 1., &simParameterMap_, &hoFilter_),
      hfAnalyzer_("HF", 1., &simParameterMap_, &hfFilter_),
      zdcAnalyzer_("ZDC", 1., &simParameterMap_, &zdcFilter_),
      hbheRecHitCollectionTag_(conf.getParameter<edm::InputTag>("hbheRecHitCollectionTag")),
      hoRecHitCollectionTag_(conf.getParameter<edm::InputTag>("hoRecHitCollectionTag")),
      hfRecHitCollectionTag_(conf.getParameter<edm::InputTag>("hfRecHitCollectionTag")) {}

namespace HcalHitAnalyzerImpl {
  template <class Collection>
  void analyze(edm::Event const &e, CaloHitAnalyzer &analyzer, edm::InputTag &tag) {
    edm::Handle<Collection> recHits;
    e.getByLabel(tag, recHits);
    for (unsigned i = 0; i < recHits->size(); ++i) {
      analyzer.analyze((*recHits)[i].id().rawId(), (*recHits)[i].energy());
    }
  }
}  // namespace HcalHitAnalyzerImpl

void HcalHitAnalyzer::analyze(edm::Event const &e, edm::EventSetup const &c) {
  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PCaloHit>> cf, zdccf;
  e.getByLabel("mix", "g4SimHitsHcalHits", cf);
  // e.getByLabel("mix", "ZDCHits", zdccf);

  // test access to SimHits for HcalHits and ZDC hits
  std::unique_ptr<MixCollection<PCaloHit>> hits(new MixCollection<PCaloHit>(cf.product()));
  // std::unique_ptr<MixCollection<PCaloHit> > zdcHits(new
  // MixCollection<PCaloHit>(zdccf.product()));
  hbheAnalyzer_.fillHits(*hits);
  // hoAnalyzer_.fillHits(*hits);
  // hfAnalyzer_.fillHits(*hits);
  // zdcAnalyzer_.fillHits(*hits);
  HcalHitAnalyzerImpl::analyze<HBHERecHitCollection>(e, hbheAnalyzer_, hbheRecHitCollectionTag_);
  HcalHitAnalyzerImpl::analyze<HORecHitCollection>(e, hoAnalyzer_, hoRecHitCollectionTag_);
  HcalHitAnalyzerImpl::analyze<HFRecHitCollection>(e, hfAnalyzer_, hfRecHitCollectionTag_);
  // HcalHitAnalyzerImpl::analyze<ZDCRecHitCollection>(e, zdcAnalyzer_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalHitAnalyzer);
