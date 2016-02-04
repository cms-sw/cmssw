#ifndef CSCALCTDigiValidation_H
#define CSCALCTDigiValidation_H

#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class CSCALCTDigiValidation : public CSCBaseValidation
{
public:
  CSCALCTDigiValidation(DQMStore* dbe, 
                        const edm::InputTag & inputTag);
  ~CSCALCTDigiValidation();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob() {}

 private:
  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theNDigisPerEventPlot;
   
};

#endif

