#ifndef CSCWireDigiValidation_H
#define CSCWireDigiValidation_H

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class CSCWireDigiValidation : public CSCBaseValidation
{
public:
  CSCWireDigiValidation(DQMStore* dbe, 
                        const edm::InputTag & inputTag,
                        bool doSim);
  ~CSCWireDigiValidation();
  void analyze(const edm::Event&, const edm::EventSetup&);

  void plotResolution(const PSimHit & hit, const CSCWireDigi & digi,
                      const CSCLayer * layer, int chamberType);

 private:
  bool theDoSimFlag;
  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theResolutionPlots[10];
  MonitorElement* theNDigisPerEventPlot;
   
};

#endif

