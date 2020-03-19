#ifndef TrackingTruthOutputTest_h
#define TrackingTruthOutputTest_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingTruthOutputTest : public edm::EDAnalyzer {
public:
  explicit TrackingTruthOutputTest(const edm::ParameterSet &conf);

  ~TrackingTruthOutputTest() override {}

  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  edm::ParameterSet conf_;
};

#endif
