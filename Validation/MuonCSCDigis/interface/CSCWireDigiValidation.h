#ifndef Validation_MuonCSCDigis_CSCWireDigiValidation_H
#define Validation_MuonCSCDigis_CSCWireDigiValidation_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCWireDigiValidation : public CSCBaseValidation {
public:
  CSCWireDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);
  ~CSCWireDigiValidation() override;
  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void plotResolution(const PSimHit &hit, const CSCWireDigi &digi, const CSCLayer *layer, int chamberType);

private:
  edm::EDGetTokenT<CSCWireDigiCollection> wires_Token_;
  edm::InputTag inputTag_;
  MonitorElement *theTimeBinPlots[10];
  MonitorElement *theNDigisPerLayerPlots[10];
  MonitorElement *theResolutionPlots[10];
  MonitorElement *theNDigisPerEventPlot;
};

#endif
