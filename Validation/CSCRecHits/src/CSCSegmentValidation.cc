#include "DQMServices/Core/interface/DQMStore.h"
#include "Validation/CSCRecHits/interface/CSCSegmentValidation.h"
#include <algorithm>

CSCSegmentValidation::CSCSegmentValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps), theLayerHitsPerChamber(), theChamberSegmentMap(), theShowerThreshold(10) {
  const auto &pset = ps.getParameterSet("cscSegment");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  segments_Token_ = iC.consumes<CSCSegmentCollection>(inputTag_);
}

CSCSegmentValidation::~CSCSegmentValidation() {}

void CSCSegmentValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  theNPerEventPlot = iBooker.book1D("CSCSegmentsPerEvent", "Number of CSC segments per event", 100, 0, 50);
  theNRecHitsPlot = iBooker.book1D("CSCRecHitsPerSegment", "Number of CSC rec hits per segment", 8, 0, 7);
  theNPerChamberTypePlot =
      iBooker.book1D("CSCSegmentsPerChamberType", "Number of CSC segments per chamber type", 11, 0, 10);
  theTypePlot4HitsNoShower = iBooker.book1D("CSCSegments4HitsNoShower", "", 100, 0, 10);
  theTypePlot4HitsNoShowerSeg = iBooker.book1D("CSCSegments4HitsNoShowerSeg", "", 100, 0, 10);
  theTypePlot4HitsShower = iBooker.book1D("CSCSegments4HitsShower", "", 100, 0, 10);
  theTypePlot4HitsShowerSeg = iBooker.book1D("CSCSegments4HitsShowerSeg", "", 100, 0, 10);
  theTypePlot5HitsNoShower = iBooker.book1D("CSCSegments5HitsNoShower", "", 100, 0, 10);
  theTypePlot5HitsNoShowerSeg = iBooker.book1D("CSCSegments5HitsNoShowerSeg", "", 100, 0, 10);
  theTypePlot5HitsShower = iBooker.book1D("CSCSegments5HitsShower", "", 100, 0, 10);
  theTypePlot5HitsShowerSeg = iBooker.book1D("CSCSegments5HitsShowerSeg", "", 100, 0, 10);
  theTypePlot6HitsNoShower = iBooker.book1D("CSCSegments6HitsNoShower", "", 100, 0, 10);
  theTypePlot6HitsNoShowerSeg = iBooker.book1D("CSCSegments6HitsNoShowerSeg", "", 100, 0, 10);
  theTypePlot6HitsShower = iBooker.book1D("CSCSegments6HitsShower", "", 100, 0, 10);
  theTypePlot6HitsShowerSeg = iBooker.book1D("CSCSegments6HitsShowerSeg", "", 100, 0, 10);

  for (int i = 1; i <= 10; ++i) {
    const std::string cn(CSCDetId::chamberName(i));

    const std::string t1("CSCSegmentRdPhiResolution_" + cn);
    const std::string t2("CSCSegmentRdPhiPull_" + cn);
    const std::string t3("CSCSegmentThetaResolution_" + cn);
    const std::string t5("CSCSegmentdXdZResolution_" + cn);
    const std::string t6("CSCSegmentdXdZPull_" + cn);
    const std::string t7("CSCSegmentdYdZResolution_" + cn);
    const std::string t8("CSCSegmentdYdZPull_" + cn);

    theRdPhiResolutionPlots[i - 1] = iBooker.book1D(t1, t1, 100, -0.4, 0.4);
    theRdPhiPullPlots[i - 1] = iBooker.book1D(t2, t2, 100, -5, 5);
    theThetaResolutionPlots[i - 1] = iBooker.book1D(t3, t3, 100, -1, 1);
    thedXdZResolutionPlots[i - 1] = iBooker.book1D(t5, t5, 100, -1, 1);
    thedXdZPullPlots[i - 1] = iBooker.book1D(t6, t6, 100, -5, 5);
    thedYdZResolutionPlots[i - 1] = iBooker.book1D(t7, t7, 100, -1, 1);
    thedYdZPullPlots[i - 1] = iBooker.book1D(t8, t8, 100, -5, 5);
  }
}

void CSCSegmentValidation::analyze(const edm::Event &e, const edm::EventSetup &eventSetup) {
  // get the collection of CSCRecHsegmentItrD
  edm::Handle<CSCSegmentCollection> hRecHits;
  e.getByToken(segments_Token_, hRecHits);
  const CSCSegmentCollection *cscRecHits = hRecHits.product();

  theChamberSegmentMap.clear();
  unsigned nPerEvent = 0;
  for (auto segmentItr = cscRecHits->begin(); segmentItr != cscRecHits->end(); segmentItr++) {
    ++nPerEvent;
    int detId = segmentItr->geographicalId().rawId();
    int chamberType = segmentItr->cscDetId().iChamberType();

    theNRecHitsPlot->Fill(segmentItr->nRecHits());
    theNPerChamberTypePlot->Fill(chamberType);
    theChamberSegmentMap[detId].push_back(*segmentItr);

    // do the resolution plots
    const PSimHit *hit = keyHit(detId);
    if (hit != nullptr) {
      const CSCLayer *layer = findLayer(hit->detUnitId());
      plotResolution(*hit, *segmentItr, layer, chamberType);
    }
  }

  theNPerEventPlot->Fill(nPerEvent);

  fillLayerHitsPerChamber();
  fillEfficiencyPlots();
}

void CSCSegmentValidation::fillEfficiencyPlots() {
  // now plot efficiency by looping over all chambers with hits
  for (auto mapItr = theLayerHitsPerChamber.begin(), mapEnd = theLayerHitsPerChamber.end(); mapItr != mapEnd;
       ++mapItr) {
    int chamberId = mapItr->first;
    int nHitsInChamber = mapItr->second.size();
    bool isShower = (nHitsInChamber > theShowerThreshold);
    bool hasSeg = hasSegment(chamberId);
    int chamberType = CSCDetId(chamberId).iChamberType();
    // find how many layers were hit in this chamber
    std::vector<int> v = mapItr->second;
    std::sort(v.begin(), v.end());
    // maybe can just count
    v.erase(std::unique(v.begin(), v.end()), v.end());
    int nLayersHit = v.size();

    if (nLayersHit == 4) {
      if (isShower)
        theTypePlot4HitsShower->Fill(chamberType);
      else
        theTypePlot4HitsNoShower->Fill(chamberType);

      if (hasSeg) {
        if (isShower)
          theTypePlot4HitsShowerSeg->Fill(chamberType);
        else
          theTypePlot4HitsNoShowerSeg->Fill(chamberType);
      }
    }

    if (nLayersHit == 5) {
      if (isShower)
        theTypePlot5HitsShower->Fill(chamberType);
      else
        theTypePlot5HitsNoShower->Fill(chamberType);

      if (hasSeg) {
        if (isShower)
          theTypePlot5HitsShowerSeg->Fill(chamberType);
        else
          theTypePlot5HitsNoShowerSeg->Fill(chamberType);
      }
    }

    if (nLayersHit == 6) {
      if (isShower)
        theTypePlot6HitsShower->Fill(chamberType);
      else
        theTypePlot6HitsNoShower->Fill(chamberType);

      if (hasSeg) {
        if (isShower)
          theTypePlot6HitsShowerSeg->Fill(chamberType);
        else
          theTypePlot6HitsNoShowerSeg->Fill(chamberType);
      }
    }
  }
}

bool CSCSegmentValidation::hasSegment(int chamberId) const {
  return (theChamberSegmentMap.find(chamberId) != theChamberSegmentMap.end());
}

void CSCSegmentValidation::plotResolution(const PSimHit &simHit,
                                          const CSCSegment &segment,
                                          const CSCLayer *layer,
                                          int chamberType) {
  GlobalPoint simHitPos = layer->toGlobal(simHit.localPosition());
  GlobalPoint segmentPos = layer->toGlobal(segment.localPosition());
  LocalVector simHitDir = simHit.localDirection();
  LocalVector segmentDir = segment.localDirection();

  double dphi = segmentPos.phi() - simHitPos.phi();
  double rdphi = segmentPos.perp() * dphi;
  double dtheta = segmentPos.theta() - simHitPos.theta();

  double sigmax = sqrt(segment.localPositionError().xx());

  double ddxdz = segmentDir.x() / segmentDir.z() - simHitDir.x() / simHitDir.z();
  double ddydz = segmentDir.y() / segmentDir.z() - simHitDir.y() / simHitDir.z();
  double sigmadxdz = sqrt(segment.localDirectionError().xx());
  double sigmadydz = sqrt(segment.localDirectionError().yy());

  theRdPhiResolutionPlots[chamberType - 1]->Fill(rdphi);
  theRdPhiPullPlots[chamberType - 1]->Fill(rdphi / sigmax);
  theThetaResolutionPlots[chamberType - 1]->Fill(dtheta);

  thedXdZResolutionPlots[chamberType - 1]->Fill(ddxdz);
  thedXdZPullPlots[chamberType - 1]->Fill(ddxdz / sigmadxdz);
  thedYdZResolutionPlots[chamberType - 1]->Fill(ddydz);
  thedYdZPullPlots[chamberType - 1]->Fill(ddydz / sigmadydz);
}

void CSCSegmentValidation::fillLayerHitsPerChamber() {
  theLayerHitsPerChamber.clear();
  std::vector<int> layersHit = theSimHitMap->detsWithHits();
  for (auto layerItr = layersHit.begin(), layersHitEnd = layersHit.end(); layerItr != layersHitEnd; ++layerItr) {
    CSCDetId layerId(*layerItr);
    CSCDetId chamberId = layerId.chamberId();
    int nhits = theSimHitMap->hits(*layerItr).size();
    // multiple entries, so we can see showers
    for (int i = 0; i < nhits; ++i) {
      theLayerHitsPerChamber[chamberId.rawId()].push_back(*layerItr);
    }
  }
}

const PSimHit *CSCSegmentValidation::keyHit(int chamberId) const {
  auto SimHitPabsLessThan = [](const PSimHit &p1, const PSimHit &p2) -> bool { return p1.pabs() < p2.pabs(); };

  const PSimHit *result = nullptr;
  int layerId = chamberId + 3;
  const auto &layerHits = theSimHitMap->hits(layerId);

  if (!layerHits.empty()) {
    // pick the hit with maximum energy
    auto hitItr = std::max_element(layerHits.begin(), layerHits.end(), SimHitPabsLessThan);
    result = &(*hitItr);
  }
  return result;
}
