#ifndef CSCDigitizer_CSCDigiDump_h
#define CSCDigitizer_CSCDigiDump_h

#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CSCDigiDump : public edm::one::EDAnalyzer<> {
public:
  explicit CSCDigiDump(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

private:
  edm::EDGetTokenT<CSCWireDigiCollection> wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_token;
  edm::EDGetTokenT<CSCComparatorDigiCollection> cd_token;
};

#endif
