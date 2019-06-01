#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/MCTruth/interface/CSCTruthTest.h"

CSCTruthTest::CSCTruthTest(const edm::ParameterSet &iConfig) : conf_(iConfig) {}

CSCTruthTest::~CSCTruthTest() {}

void CSCTruthTest::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  Handle<CSCRecHit2DCollection> cscRecHits;
  iEvent.getByLabel("csc2DRecHits", cscRecHits);

  MuonTruth theTruth(iEvent, iSetup, conf_);

  for (CSCRecHit2DCollection::const_iterator recHitItr = cscRecHits->begin(); recHitItr != cscRecHits->end();
       recHitItr++) {
    theTruth.analyze(*recHitItr);
    edm::LogVerbatim("SimMuonCSCTruthTest") << theTruth.muonFraction() << " " << recHitItr->cscDetId();
  }
}
