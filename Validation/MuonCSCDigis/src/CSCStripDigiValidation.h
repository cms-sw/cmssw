#ifndef CSCStripDigiValidation_H
#define CSCStripDigiValidation_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "Validation/MuonCSCDigis/src/PSimHitMap.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

class CSCStripDigiValidation
{
public:
  CSCStripDigiValidation(const edm::ParameterSet&, DaqMonitorBEInterface* dbe, 
                         const PSimHitMap & hitMap);
  ~CSCStripDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&) {}
  void endJob() {}
 
  void setGeometry(const CSCGeometry * geom) {theCSCGeometry = geom;}

  void plotResolution(const PSimHit & hit, int strip,
                      const CSCLayer * layer, int chamberType);

  const CSCLayer * findLayer(int detId) const;

 private:
  void fillPedestalPlots(const CSCStripDigi & digi);
  void fillSignalPlots(const CSCStripDigi & digi);

  DaqMonitorBEInterface* dbe_;
  edm::InputTag theInputTag;
  const PSimHitMap & theSimHitMap;
  const CSCGeometry * theCSCGeometry;

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
  MonitorElement* theResolutionPlots[10];   
};

#endif

