#ifndef Validation_MuonCSCDigis_CSCCorrelatedLCTDigiValidation_H
#define Validation_MuonCSCDigis_CSCCorrelatedLCTDigiValidation_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCCorrelatedLCTDigiValidation : public CSCBaseValidation {
public:
  CSCCorrelatedLCTDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);
  ~CSCCorrelatedLCTDigiValidation() override;
  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> lcts_Token_;
  edm::InputTag inputTag_;
  MonitorElement *theNDigisPerChamberPlots[10];
  MonitorElement *theNDigisPerEventPlot;

  // more diagnostic plots
  std::vector<std::string> chambers_;
  std::vector<unsigned> chambersRun3_;

  std::vector<std::string> lctVars_;
  std::vector<unsigned> lctNBin_;
  std::vector<double> lctMinBin_;
  std::vector<double> lctMaxBin_;

  bool isRun3_;

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement *> > chamberHistos;
};

#endif
