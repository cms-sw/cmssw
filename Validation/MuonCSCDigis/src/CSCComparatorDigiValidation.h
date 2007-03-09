#ifndef CSCComparatorDigiValidation_H
#define CSCComparatorDigiValidation_H

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCComparatorDigiValidation : public CSCBaseValidation
{
public:
  CSCComparatorDigiValidation(DaqMonitorBEInterface* dbe, const edm::InputTag & inputTag);
  ~CSCComparatorDigiValidation();
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theNDigisPerEventPlot;

 
};

#endif

