#ifndef MCTruth_CSCTruthTest_h
#define MCTruth_CSCTruthTest_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "SimMuon/MCTruth/interface/MuonTruth.h"

class CSCTruthTest : public edm::stream::EDAnalyzer<> {
public:
  explicit CSCTruthTest(const edm::ParameterSet &);
  ~CSCTruthTest() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  const edm::ParameterSet &conf_;
};

#endif
