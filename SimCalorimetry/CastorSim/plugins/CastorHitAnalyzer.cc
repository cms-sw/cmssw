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
  const edm::EDGetTokenT<CastorRecHitCollection> castorRecHitToken_;
  const edm::EDGetTokenT<CrossingFrame<PCaloHit>> castorcfToken_;
};

CastorHitAnalyzer::CastorHitAnalyzer(edm::ParameterSet const &conf)
    : hitReadoutName_("CastorHits"),
      simParameterMap_(),
      castorFilter_(),
      castorAnalyzer_("CASTOR", 1., &simParameterMap_, &castorFilter_),
      castorRecHitToken_(
          consumes<CastorRecHitCollection>(conf.getParameter<edm::InputTag>("castorRecHitCollectionTag"))),
      castorcfToken_(consumes<CrossingFrame<PCaloHit>>(edm::InputTag("mix", "g4SimHitsCastorFI"))) {}

namespace CastorHitAnalyzerImpl {
  template <class Collection>
  void analyze(edm::Event const &e, CaloHitAnalyzer &analyzer, const edm::EDGetTokenT<Collection> &token) {
    const edm::Handle<Collection> &recHits = e.getHandle(token);
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
  const edm::Handle<CrossingFrame<PCaloHit>> &castorcf = e.getHandle(castorcfToken_);

  // access to SimHits
  std::unique_ptr<MixCollection<PCaloHit>> hits(new MixCollection<PCaloHit>(castorcf.product()));
  castorAnalyzer_.fillHits(*hits);
  CastorHitAnalyzerImpl::analyze<CastorRecHitCollection>(e, castorAnalyzer_, castorRecHitToken_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CastorHitAnalyzer);
