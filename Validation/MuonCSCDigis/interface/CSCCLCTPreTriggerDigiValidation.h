#ifndef Validation_MuonCSCDigis_CSCCLCTPreTriggerDigiValidation_H
#define Validation_MuonCSCDigis_CSCCLCTPreTriggerDigiValidation_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigiCollection.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCCLCTPreTriggerDigiValidation : public CSCBaseValidation {
public:
  CSCCLCTPreTriggerDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);
  ~CSCCLCTPreTriggerDigiValidation() override;
  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<CSCCLCTPreTriggerDigiCollection> preclcts_Token_;
  edm::InputTag inputTag_;

  // more diagnostic plots
  std::vector<std::string> chambers_;
  std::vector<unsigned> chambersRun3_;

  std::vector<std::string> preclctVars_;
  std::vector<unsigned> preclctNBin_;
  std::vector<double> preclctMinBin_;
  std::vector<double> preclctMaxBin_;

  bool isRun3_;

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement *> > chamberHistos;
};

#endif
