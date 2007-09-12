

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/MCTruth/interface/MuonTruth.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"

//
// class decleration
//

class CSCTruthTest : public edm::EDAnalyzer {
public:
  explicit CSCTruthTest(const edm::ParameterSet&);
  ~CSCTruthTest();


private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  MuonTruth theTruth;  

      // ----------member data ---------------------------
};

CSCTruthTest::CSCTruthTest(const edm::ParameterSet& iConfig)
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

  theTruth.eventSetup(iEvent);

  for(CSCRecHit2DCollection::const_iterator recHitItr = cscRecHits->begin();
      recHitItr != cscRecHits->end(); recHitItr++)
  {
     theTruth.analyze(*recHitItr);
     std::cout << theTruth.muonFraction() << " " << recHitItr->cscDetId() << std::endl;
  }
}


