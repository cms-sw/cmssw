#ifndef CSCSegmentValidation_h
#define CSCSegmentValidation_h

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"

class CSCSegmentValidation : public CSCBaseValidation
{
public:
  CSCSegmentValidation(DaqMonitorBEInterface* dbe, const edm::InputTag & inputTag);

  virtual ~CSCSegmentValidation() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  void plotResolution(const PSimHit & simHit, const CSCSegment & recHit,
                      const CSCLayer * layer, int chamberType);

  MonitorElement* theNPerEventPlot;
  MonitorElement* theNRecHitsPlot;
  MonitorElement* theNPerChamberTypePlot;
  MonitorElement* theResolutionPlots[10];
  MonitorElement* thePullPlots[10];
};

#endif

