#ifndef Validation_MuonCSCDigis_CSCALCTDigiValidation_H
#define Validation_MuonCSCDigis_CSCALCTDigiValidation_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCALCTDigiValidation : public CSCBaseValidation {
public:
  CSCALCTDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);
  ~CSCALCTDigiValidation() override;
  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<CSCALCTDigiCollection> alcts_Token_;
  edm::InputTag inputTag_;
  MonitorElement *theNDigisPerChamberPlots[10];
  MonitorElement *theNDigisPerEventPlot;

  // more diagnostic plots
  std::vector<std::string> chambers_;

  std::vector<std::string> alctVars_;
  std::vector<unsigned> alctNBin_;
  std::vector<double> alctMinBin_;
  std::vector<double> alctMaxBin_;

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement *> > chamberHistos;
};

#endif
