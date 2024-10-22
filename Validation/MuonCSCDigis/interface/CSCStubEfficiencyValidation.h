#ifndef Validation_MuonCSCDigis_CSCStubEfficiencyValidation_H
#define Validation_MuonCSCDigis_CSCStubEfficiencyValidation_H

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

#include <string>

class CSCStubMatcher;

class CSCStubEfficiencyValidation : public CSCBaseValidation {
public:
  CSCStubEfficiencyValidation(const edm::ParameterSet &pset, edm::ConsumesCollector &&iC);
  ~CSCStubEfficiencyValidation() override;

  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  std::unique_ptr<CSCStubMatcher> cscStubMatcher_;

  // efficiency vs eta for each CSC TP
  MonitorElement *etaALCTNum[10];
  MonitorElement *etaALCTDenom[10];
  MonitorElement *etaCLCTNum[10];
  MonitorElement *etaCLCTDenom[10];
  MonitorElement *etaLCTNum[10];
  MonitorElement *etaLCTDenom[10];

  edm::EDGetTokenT<edm::SimVertexContainer> simVertexInput_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackInput_;

  std::vector<double> etaMins_;
  std::vector<double> etaMaxs_;
};

#endif
