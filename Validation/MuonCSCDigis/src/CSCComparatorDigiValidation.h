#ifndef CSCComparatorDigiValidation_H
#define CSCComparatorDigiValidation_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "Validation/MuonCSCDigis/src/PSimHitMap.h"

class CSCComparatorDigiValidation
{
public:
  CSCComparatorDigiValidation(const edm::ParameterSet&, DaqMonitorBEInterface* dbe,
                              const PSimHitMap & hitMap);
  ~CSCComparatorDigiValidation();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&) {}
  void endJob() {}
 

 private:
  DaqMonitorBEInterface* dbe_;
  edm::InputTag theInputTag;
  const PSimHitMap & theSimHitMap;

  MonitorElement* theTimeBinPlots[10];
  MonitorElement* theNDigisPerLayerPlots[10];
  MonitorElement* theNDigisPerEventPlot;

 
};

#endif

