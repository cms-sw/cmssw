#ifndef CSCComparatorDigiValidation_H
#define CSCComparatorDigiValidation_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class CSCComparatorDigiValidation : public CSCBaseValidation
{
public:
  CSCComparatorDigiValidation(const edm::InputTag & inputTag,
                              const edm::InputTag & stripDigiInputTag,
                              edm::ConsumesCollector && iC);
  ~CSCComparatorDigiValidation();
  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  edm::EDGetTokenT<CSCStripDigiCollection> theStripDigi_Token_;
  edm::EDGetTokenT<CSCComparatorDigiCollection> comparators_Token_;

  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theStripDigiPlots[10];
  MonitorElement* the3StripPlots[10];

  MonitorElement* theNDigisPerEventPlot;
};

#endif
