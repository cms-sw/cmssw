#ifndef CSCStripDigiValidation_H
#define CSCStripDigiValidation_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"

class CSCStripDigiValidation
{
public:
  CSCStripDigiValidation(const edm::ParameterSet&, DaqMonitorBEInterface* dbe);
  ~CSCStripDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&) {}
  void endJob() {}
 

 private:
  void fillPedestalPlots(const CSCStripDigi & digi);
  void fillSignalPlots(const CSCStripDigi & digi);

  DaqMonitorBEInterface* dbe_;
  edm::InputTag theInputTag;
  float thePedestalSum;
  float thePedestalCovarianceSum;
  int thePedestalCount;

  MonitorElement* thePedestalPlot;
  MonitorElement* thePedestalTimeCorrelationPlot;
  MonitorElement* thePedestalNeighborCorrelationPlot;
  MonitorElement* theAmplitudePlot;
  MonitorElement* theRatio4to5Plot;
  MonitorElement* theRatio6to5Plot;
  MonitorElement* theNDigisPerLayerPlot;
  MonitorElement* theNDigisPerChamberPlot;
  MonitorElement* theNDigisPerEventPlot;
   
};

#endif

