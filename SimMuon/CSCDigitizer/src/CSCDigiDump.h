#ifndef CSCDigitizer_CSCDigiDump_h
#define CSCDigitizer_CSCDigiDump_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

class CSCDigiDump : public edm::EDAnalyzer {
public:
  explicit CSCDigiDump(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

private:
  edm::InputTag wireDigiTag_;
  edm::InputTag stripDigiTag_;
  edm::InputTag comparatorDigiTag_;
};

#endif

