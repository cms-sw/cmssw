#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/CSCRecHits/interface/CSCRecHit2DValidation.h"

CSCRecHit2DValidation::CSCRecHit2DValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps), theNPerEventPlot(nullptr) {
  const auto &pset = ps.getParameterSet("cscRecHit");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  rechits_Token_ = iC.consumes<CSCRecHit2DCollection>(inputTag_);
}

CSCRecHit2DValidation::~CSCRecHit2DValidation() {}

void CSCRecHit2DValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  theNPerEventPlot = iBooker.book1D("CSCRecHitsPerEvent", "Number of CSC Rec Hits per event", 100, 0, 500);
  // 10 chamber types, if you consider ME1/a and ME1/b separate
  for (int i = 1; i <= 10; ++i) {
    const std::string cn(CSCDetId::chamberName(i));
    const std::string t1("CSCRecHitResolution_" + cn);
    const std::string t2("CSCRecHitPull_" + cn);
    const std::string t3("CSCRecHitYResolution_" + cn);

    const std::string t4("CSCRecHitYPull_" + cn);
    const std::string t5("CSCRecHitPosInStrip_" + cn);
    const std::string t6("CSCSimHitPosInStrip_" + cn);

    const std::string t7("CSCRecHit_" + cn);
    const std::string t8("CSCSimHit_" + cn);
    const std::string t9("CSCTPeak_" + cn);

    theResolutionPlots[i - 1] = iBooker.book1D(t1, t1 + ";R*dPhi Resolution [cm];Entries", 100, -0.2, 0.2);
    thePullPlots[i - 1] = iBooker.book1D(t2, t2 + ";dPhi Pull;Entries", 100, -3, 3);
    theYResolutionPlots[i - 1] = iBooker.book1D(t3, t3 + ";Local Y Resolution [cm];Entries", 100, -5, 5);
    theYPullPlots[i - 1] = iBooker.book1D(t4, t4 + ";Local Y Pull;Entries", 100, -3, 3);
    theRecHitPosInStrip[i - 1] = iBooker.book1D(t5, t5 + ";Position in Strip;Entries", 100, -2, 2);
    theSimHitPosInStrip[i - 1] = iBooker.book1D(t6, t6 + ";Position in Strip;Entries", 100, -2, 2);

    theScatterPlots[i - 1] = iBooker.book2D(t7, t7 + ";Local Phi;Local Y [cm]", 200, -20, 20, 200, -250, 250);
    theSimHitScatterPlots[i - 1] = iBooker.book2D(t8, t8 + ";Local Phi;Local Y [cm]", 200, -20, 20, 200, -250, 250);
    theTPeaks[i - 1] = iBooker.book1D(t9, t9 + ";Peak Time [ns];Entries", 200, 0, 400);
  }
}

void CSCRecHit2DValidation::analyze(const edm::Event &e, const edm::EventSetup &eventSetup) {
  // get the collection of CSCRecHrecHitItrD
  edm::Handle<CSCRecHit2DCollection> hRecHits;
  e.getByToken(rechits_Token_, hRecHits);
  const CSCRecHit2DCollection *cscRecHits = hRecHits.product();

  unsigned nPerEvent = 0;

  for (auto recHitItr = cscRecHits->begin(); recHitItr != cscRecHits->end(); recHitItr++) {
    ++nPerEvent;
    int detId = (*recHitItr).cscDetId().rawId();
    edm::PSimHitContainer simHits = theSimHitMap->hits(detId);
    const CSCLayer *layer = findLayer(detId);
    int chamberType = layer->chamber()->specs()->chamberType();
    theTPeaks[chamberType - 1]->Fill(recHitItr->tpeak());
    if (simHits.size() == 1) {
      plotResolution(simHits[0], *recHitItr, layer, chamberType);
    }
    float localX = recHitItr->localPosition().x();
    float localY = recHitItr->localPosition().y();
    // find a local phi
    float globalR = layer->toGlobal(LocalPoint(0., 0., 0.)).perp();
    GlobalPoint axisThruChamber(globalR + localY, localX, 0.);
    float localPhi = axisThruChamber.phi().degrees();
    theScatterPlots[chamberType - 1]->Fill(localPhi, localY);
  }
  theNPerEventPlot->Fill(nPerEvent);

  if (doSim_) {
    // fill sim hits
    std::vector<int> layersWithSimHits = theSimHitMap->detsWithHits();
    for (unsigned i = 0; i < layersWithSimHits.size(); ++i) {
      edm::PSimHitContainer simHits = theSimHitMap->hits(layersWithSimHits[i]);
      for (auto hitItr = simHits.begin(); hitItr != simHits.end(); ++hitItr) {
        const CSCLayer *layer = findLayer(layersWithSimHits[i]);
        int chamberType = layer->chamber()->specs()->chamberType();
        float localX = hitItr->localPosition().x();
        float localY = hitItr->localPosition().y();
        // find a local phi
        float globalR = layer->toGlobal(LocalPoint(0., 0., 0.)).perp();
        GlobalPoint axisThruChamber(globalR + localY, localX, 0.);
        float localPhi = axisThruChamber.phi().degrees();
        theSimHitScatterPlots[chamberType - 1]->Fill(localPhi, localY);
      }
    }
  }
}

void CSCRecHit2DValidation::plotResolution(const PSimHit &simHit,
                                           const CSCRecHit2D &recHit,
                                           const CSCLayer *layer,
                                           int chamberType) {
  GlobalPoint simHitPos = layer->toGlobal(simHit.localPosition());
  GlobalPoint recHitPos = layer->toGlobal(recHit.localPosition());

  double dphi = recHitPos.phi() - simHitPos.phi();
  double rdphi = recHitPos.perp() * dphi;
  theResolutionPlots[chamberType - 1]->Fill(rdphi);
  thePullPlots[chamberType - 1]->Fill(rdphi / sqrt(recHit.localPositionError().xx()));
  double dy = recHit.localPosition().y() - simHit.localPosition().y();
  theYResolutionPlots[chamberType - 1]->Fill(dy);
  theYPullPlots[chamberType - 1]->Fill(dy / sqrt(recHit.localPositionError().yy()));

  const CSCLayerGeometry *layerGeometry = layer->geometry();
  float recStrip = layerGeometry->strip(recHit.localPosition());
  float simStrip = layerGeometry->strip(simHit.localPosition());
  theRecHitPosInStrip[chamberType - 1]->Fill(recStrip - int(recStrip));
  theSimHitPosInStrip[chamberType - 1]->Fill(simStrip - int(simStrip));
}
