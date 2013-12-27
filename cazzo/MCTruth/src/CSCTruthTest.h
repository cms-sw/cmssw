// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimMuon/MCTruth/interface/MuonTruth.h"


class CSCTruthTest : public edm::EDAnalyzer {
public:
  explicit CSCTruthTest(const edm::ParameterSet&);
  ~CSCTruthTest();


private:

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  const edm::ParameterSet& conf_;

};

