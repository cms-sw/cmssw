#ifndef Validation_MuonCSCDigis_CSCCLCTDigiValidation_H
#define Validation_MuonCSCDigis_CSCCLCTDigiValidation_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCCLCTDigiValidation : public CSCBaseValidation {
public:
  CSCCLCTDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);
  ~CSCCLCTDigiValidation() override;
  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<CSCCLCTDigiCollection> clcts_Token_;
  edm::InputTag inputTag_;
  MonitorElement *theNDigisPerChamberPlots[10];
  MonitorElement *theNDigisPerEventPlot;

  // more diagnostic plots
  std::vector<std::string> chambers_;
  std::vector<unsigned> chambersRun3_;

  std::vector<std::string> clctVars_;
  std::vector<unsigned> clctNBin_;
  std::vector<double> clctMinBin_;
  std::vector<double> clctMaxBin_;

  bool isRun3_;

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement *> > chamberHistos;
};

#endif
