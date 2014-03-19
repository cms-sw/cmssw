#ifndef CSCSegmentValidation_h
#define CSCSegmentValidation_h

#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class CSCSegmentValidation : public CSCBaseValidation
{
public:
  CSCSegmentValidation(DQMStore* dbe, const edm::InputTag & inputTag, edm::ConsumesCollector && iC);

  virtual ~CSCSegmentValidation() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  void plotResolution(const PSimHit & simHit, const CSCSegment & recHit,
                      const CSCLayer * layer, int chamberType);

  bool hasSegment(int chamberId) const;
  static int whatChamberType(int detId);

  edm::EDGetTokenT<CSCSegmentCollection> segments_Token_;

  // map to count how many layers are hit.  First index is chamber detId, second is layers
  // that have hits
  typedef std::map<int, std::vector<int> > ChamberHitMap;
  ChamberHitMap theLayerHitsPerChamber;
  void fillLayerHitsPerChamber();
  void fillEfficiencyPlots();
  // decide which SimHit to use for the direction
  const PSimHit * keyHit(int chamberId) const;

  typedef std::map<int, std::vector<CSCSegment> > ChamberSegmentMap;
  ChamberSegmentMap theChamberSegmentMap;
  // the number of hits in a chamber to make it a shower
  int theShowerThreshold;

  MonitorElement* theNPerEventPlot;
  MonitorElement* theNRecHitsPlot;
  MonitorElement* theNPerChamberTypePlot;
  MonitorElement* theRdPhiResolutionPlots[10];
  MonitorElement* theRdPhiPullPlots[10];
  MonitorElement* theThetaResolutionPlots[10];
  MonitorElement* theThetaPullPlots[10];
  MonitorElement* thedXdZResolutionPlots[10];
  MonitorElement* thedXdZPullPlots[10];
  MonitorElement* thedYdZResolutionPlots[10];
  MonitorElement* thedYdZPullPlots[10];



  MonitorElement* theTypePlot4HitsNoShower;
  MonitorElement* theTypePlot4HitsNoShowerSeg;
  MonitorElement* theTypePlot4HitsShower;
  MonitorElement* theTypePlot4HitsShowerSeg;
  MonitorElement* theTypePlot5HitsNoShower;
  MonitorElement* theTypePlot5HitsNoShowerSeg;
  MonitorElement* theTypePlot5HitsShower;
  MonitorElement* theTypePlot5HitsShowerSeg;
  MonitorElement* theTypePlot6HitsNoShower;
  MonitorElement* theTypePlot6HitsNoShowerSeg;
  MonitorElement* theTypePlot6HitsShower;
  MonitorElement* theTypePlot6HitsShowerSeg;


};

#endif

