/** Compares RecHits to SimHit

  \Author P. Katsas, Univ. of Athens
*/

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/CastorSim/interface/CastorHitFilter.h"
#include "SimCalorimetry/CastorSim/interface/CastorSimParameterMap.h"

#include <iostream>
#include <string>

class CastorHitAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CastorHitAnalyzer(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

private:
  std::string hitReadoutName_;
  CastorSimParameterMap simParameterMap_;
  CastorHitFilter castorFilter_;
  CaloHitAnalyzer castorAnalyzer_;
  edm::InputTag castorRecHitCollectionTag_;
};

CastorHitAnalyzer::CastorHitAnalyzer(edm::ParameterSet const &conf)
    : hitReadoutName_("CastorHits"),
      simParameterMap_(),
      castorFilter_(),
      castorAnalyzer_("CASTOR", 1., &simParameterMap_, &castorFilter_),
      castorRecHitCollectionTag_(conf.getParameter<edm::InputTag>("castorRecHitCollectionTag")) {}

namespace CastorHitAnalyzerImpl {
  template <class Collection>
  void analyze(edm::Event const &e, CaloHitAnalyzer &analyzer, edm::InputTag &tag) {
    edm::Handle<Collection> recHits;
    e.getByLabel(tag, recHits);
    if (!recHits.isValid()) {
      edm::LogError("CastorHitAnalyzer") << "Could not find Castor RecHitContainer ";
    } else {
      for (unsigned i = 0; i < recHits->size(); ++i) {
        analyzer.analyze((*recHits)[i].id().rawId(), (*recHits)[i].energy());
      }
    }
  }
}  // namespace CastorHitAnalyzerImpl

void CastorHitAnalyzer::analyze(edm::Event const &e, edm::EventSetup const &c) {
  edm::Handle<CrossingFrame<PCaloHit>> castorcf;
  e.getByLabel("mix", "g4SimHitsCastorFI", castorcf);

  // access to SimHits
  std::unique_ptr<MixCollection<PCaloHit>> hits(new MixCollection<PCaloHit>(castorcf.product()));
  castorAnalyzer_.fillHits(*hits);
  CastorHitAnalyzerImpl::analyze<CastorRecHitCollection>(e, castorAnalyzer_, castorRecHitCollectionTag_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CastorHitAnalyzer);
