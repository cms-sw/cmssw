#ifndef CSCCLCTDigiValidation_H
#define CSCCLCTDigiValidation_H

#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class CSCCLCTDigiValidation : public CSCBaseValidation
{
public:
  CSCCLCTDigiValidation(DQMStore* dbe, 
                        const edm::InputTag & inputTag);
  ~CSCCLCTDigiValidation();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob() {}

 private:
  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theNDigisPerEventPlot;
   
};

#endif

