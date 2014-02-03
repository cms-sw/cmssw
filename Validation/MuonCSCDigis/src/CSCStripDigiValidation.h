#ifndef CSCStripDigiValidation_H
#define CSCStripDigiValidation_H

#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class CSCStripDigiValidation : public CSCBaseValidation
{
public:
  CSCStripDigiValidation(DQMStore* dbe,  
                         const edm::InputTag & inputTag, bool doSim);
  ~CSCStripDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);
 
  void setGeometry(const CSCGeometry * geom) {theCSCGeometry = geom;}

  void plotResolution(const PSimHit & hit, int strip,
                      const CSCLayer * layer, int chamberType);

 private:
  void fillPedestalPlots(const CSCStripDigi & digi);
  void fillSignalPlots(const CSCStripDigi & digi);

  float thePedestalSum;
  float thePedestalCovarianceSum;
  int thePedestalCount;
  bool theDoSimFlag;
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

