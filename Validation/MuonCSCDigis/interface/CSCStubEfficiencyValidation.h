#ifndef Validation_MuonCSCDigis_CSCStubEfficiencyValidation_H
#define Validation_MuonCSCDigis_CSCStubEfficiencyValidation_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStubMatcher.h"

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

#include <map>
#include <string>
#include <tuple>

class CSCStubEfficiencyValidation : public CSCBaseValidation {
public:
  CSCStubEfficiencyValidation(const edm::ParameterSet &pset, edm::ConsumesCollector &&iC);
  ~CSCStubEfficiencyValidation() override;

  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  // access to the matcher
  std::shared_ptr<CSCStubMatcher> cscStubMatcher() { return cscStubMatcher_; }
  void setCSCStubMatcher(std::shared_ptr<CSCStubMatcher> s) { cscStubMatcher_ = s; }

private:
  bool isSimTrackGood(const SimTrack &t);

  edm::EDGetTokenT<CSCALCTDigiCollection> alcts_Token_;
  edm::EDGetTokenT<CSCCLCTDigiCollection> clcts_Token_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> lcts_Token_;
  edm::InputTag inputTag_;

  std::shared_ptr<CSCStubMatcher> cscStubMatcher_;

  // efficiency vs eta for each CSC TP
  MonitorElement *etaALCTNum[10];
  MonitorElement *etaALCTDenom[10];
  MonitorElement *etaCLCTNum[10];
  MonitorElement *etaCLCTDenom[10];
  MonitorElement *etaLCTNum[10];
  MonitorElement *etaLCTDenom[10];

  edm::EDGetTokenT<edm::SimVertexContainer> simVertexInput_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackInput_;
  double simTrackMinPt_;
  double simTrackMinEta_;
  double simTrackMaxEta_;

  std::vector<double> etaMins_;
  std::vector<double> etaMaxs_;
};

#endif
