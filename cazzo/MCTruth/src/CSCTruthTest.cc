#include "SimMuon/MCTruth/src/CSCTruthTest.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"


CSCTruthTest::CSCTruthTest(const edm::ParameterSet& iConfig):
  conf_(iConfig)
{

}


CSCTruthTest::~CSCTruthTest()
{
 
}

void
CSCTruthTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  Handle<CSCRecHit2DCollection> cscRecHits;
  iEvent.getByLabel("csc2DRecHits",cscRecHits);

  MuonTruth theTruth(iEvent,iSetup,conf_);

  for(CSCRecHit2DCollection::const_iterator recHitItr = cscRecHits->begin();
      recHitItr != cscRecHits->end(); recHitItr++)
  {
     theTruth.analyze(*recHitItr);
     std::cout << theTruth.muonFraction() << " " << recHitItr->cscDetId() << std::endl;
  }
}


