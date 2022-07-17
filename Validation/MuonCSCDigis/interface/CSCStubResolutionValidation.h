#ifndef Validation_MuonCSCDigis_CSCStubResolutionValidation_H
#define Validation_MuonCSCDigis_CSCStubResolutionValidation_H

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

#include <string>

class CSCStubMatcher;

class CSCStubResolutionValidation : public CSCBaseValidation {
public:
  CSCStubResolutionValidation(const edm::ParameterSet &pset, edm::ConsumesCollector &&iC);
  ~CSCStubResolutionValidation() override;

  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  std::unique_ptr<CSCStubMatcher> cscStubMatcher_;

  // resolution for each CSC TP; 10 CSC stations;
  MonitorElement *posresCLCT_hs[10];
  MonitorElement *posresCLCT_qs[10];
  MonitorElement *posresCLCT_es[10];

  MonitorElement *bendresCLCT[10];

  edm::EDGetTokenT<edm::SimVertexContainer> simVertexInput_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackInput_;
};

#endif
