#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/MCTruth/interface/CSCTruthTest.h"

CSCTruthTest::CSCTruthTest(const edm::ParameterSet &iConfig)
    : theTruth_(iConfig, consumesCollector()),
      cscRecHitToken_(consumes<CSCRecHit2DCollection>(edm::InputTag("csc2DRecHits"))) {}

void CSCTruthTest::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  const edm::Handle<CSCRecHit2DCollection> &cscRecHits = iEvent.getHandle(cscRecHitToken_);

  theTruth_.initEvent(iEvent, iSetup);

  for (CSCRecHit2DCollection::const_iterator recHitItr = cscRecHits->begin(); recHitItr != cscRecHits->end();
       recHitItr++) {
    theTruth_.analyze(*recHitItr);
    edm::LogVerbatim("SimMuonCSCTruthTest") << theTruth_.muonFraction() << " " << recHitItr->cscDetId();
  }
}
