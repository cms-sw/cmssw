#ifndef CSCRecHit2DValidation_h
#define CSCRecHit2DValidation_h

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"

class CSCRecHit2DValidation : public CSCBaseValidation
{
public:
  CSCRecHit2DValidation(DaqMonitorBEInterface* dbe, const edm::InputTag & inputTag);

  virtual ~CSCRecHit2DValidation() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  void plotResolution(const PSimHit & simHit, const CSCRecHit2D & recHit,
                      const CSCLayer * layer, int chamberType);

  MonitorElement* theNPerEventPlot;
  MonitorElement* theResolutionPlots[10];
  MonitorElement* thePullPlots[10];
  MonitorElement* theYResolutionPlots[10];
  MonitorElement* theYPullPlots[10];
  MonitorElement* thePhiPlots[10];
  MonitorElement* theYPlots[10];
};

#endif

