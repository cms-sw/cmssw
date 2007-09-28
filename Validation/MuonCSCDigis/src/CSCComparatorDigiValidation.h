#ifndef CSCComparatorDigiValidation_H
#define CSCComparatorDigiValidation_H

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCComparatorDigiValidation : public CSCBaseValidation
{
public:
  CSCComparatorDigiValidation(DaqMonitorBEInterface* dbe, 
    const edm::InputTag & inputTag, const edm::InputTag & stripDigiInputTag);
  ~CSCComparatorDigiValidation();
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag theStripDigiInputTag;

  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theStripDigiPlots[10];
  MonitorElement* the3StripPlots[10];

  MonitorElement* theNDigisPerEventPlot;

 
};

#endif

