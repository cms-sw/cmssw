#ifndef CSCWireDigiValidation_H
#define CSCWireDigiValidation_H

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCWireDigiValidation : public CSCBaseValidation
{
public:
  CSCWireDigiValidation(DaqMonitorBEInterface* dbe, const edm::InputTag & inputTag);
  ~CSCWireDigiValidation();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&) {}
  void endJob() {}

  void plotResolution(const PSimHit & hit, const CSCWireDigi & digi,
                      const CSCLayer * layer, int chamberType);

 private:

  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theResolutionPlots[10];
  MonitorElement* theNDigisPerEventPlot;
   
};

#endif

