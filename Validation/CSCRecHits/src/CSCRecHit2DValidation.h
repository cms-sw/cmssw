#ifndef CSCRecHit2DValidation_h
#define CSCRecHit2DValidation_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

class CSCRecHit2DValidation : public CSCBaseValidation {
public:
  CSCRecHit2DValidation(const edm::InputTag &inputTag, edm::ConsumesCollector &&iC);
  ~CSCRecHit2DValidation() override;
  void bookHistograms(DQMStore::IBooker &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<CSCRecHit2DCollection> rechits_Token_;

  void plotResolution(const PSimHit &simHit, const CSCRecHit2D &recHit, const CSCLayer *layer, int chamberType);

  MonitorElement *theNPerEventPlot;
  MonitorElement *theResolutionPlots[10];
  MonitorElement *thePullPlots[10];
  MonitorElement *theYResolutionPlots[10];
  MonitorElement *theYPullPlots[10];
  MonitorElement *theScatterPlots[10];
  MonitorElement *theSimHitScatterPlots[10];
  MonitorElement *theRecHitPosInStrip[10];
  MonitorElement *theSimHitPosInStrip[10];
  MonitorElement *theTPeaks[10];
};

#endif
