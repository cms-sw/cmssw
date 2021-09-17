#ifndef Validation_MuonCSCDigis_CSCALCTDigiValidation_H
#define Validation_MuonCSCDigis_CSCALCTDigiValidation_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
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
  MonitorElement *theTimeBinPlots[10];
  MonitorElement *theNDigisPerChamberPlots[10];
  MonitorElement *theNDigisPerEventPlot;
};

#endif
