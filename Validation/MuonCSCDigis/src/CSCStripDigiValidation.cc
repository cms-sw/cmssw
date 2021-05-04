#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCStripDigiValidation.h"

CSCStripDigiValidation::CSCStripDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps),
      thePedestalSum(0),
      thePedestalCovarianceSum(0),
      thePedestalCount(0),
      thePedestalTimeCorrelationPlot(nullptr),
      thePedestalNeighborCorrelationPlot(nullptr),
      theNDigisPerChamberPlot(nullptr) {
  const auto &pset = ps.getParameterSet("cscStripDigi");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  strips_Token_ = iC.consumes<CSCStripDigiCollection>(inputTag_);
}

CSCStripDigiValidation::~CSCStripDigiValidation() {}

void CSCStripDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  thePedestalPlot = iBooker.book1D("CSCPedestal", "CSC Pedestal;ADC Counts;Entries", 400, 550, 650);
  theAmplitudePlot = iBooker.book1D("CSCStripAmplitude", "CSC Strip Amplitude;Strip Amplitude;Entries", 200, 0, 2000);
  theRatio4to5Plot = iBooker.book1D("CSCStrip4to5", "CSC Strip Ratio tbin 4 to tbin 5;Strip Ratio;Entries", 100, 0, 1);
  theRatio6to5Plot =
      iBooker.book1D("CSCStrip6to5", "CSC Strip Ratio tbin 6 to tbin 5;Strip Ratio;Entries", 120, 0, 1.2);
  theNDigisPerLayerPlot =
      iBooker.book1D("CSCStripDigisPerLayer",
                     "Number of CSC Strip Digis per layer;Number of CSC Strip Digis per layer;Entries",
                     48,
                     0,
                     48);
  theNDigisPerEventPlot =
      iBooker.book1D("CSCStripDigisPerEvent",
                     "Number of CSC Strip Digis per event;Number of CSC Strip Digis per event;Entries",
                     100,
                     0,
                     500);

  if (doSim_) {
    for (int i = 1; i <= 10; ++i) {
      const std::string t1("CSCStripPosResolution_" + CSCDetId::chamberName(i));
      theResolutionPlots[i - 1] = iBooker.book1D(
          t1,
          "Strip X Position Resolution " + CSCDetId::chamberName(i) + ";Strip X Position Resolution; Entries",
          100,
          -5,
          5);
    }
  }
}

void CSCStripDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &) {
  edm::Handle<CSCStripDigiCollection> strips;
  e.getByToken(strips_Token_, strips);
  if (!strips.isValid()) {
    edm::LogError("CSCStripDigiValidation") << "Cannot get strips by label " << inputTag_.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (auto j = strips->begin(); j != strips->end(); j++) {
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    int detId = (*j).first.rawId();

    const CSCLayer *layer = findLayer(detId);
    int chamberType = layer->chamber()->specs()->chamberType();
    int nDigis = last - digiItr;
    nDigisPerEvent += nDigis;
    theNDigisPerLayerPlot->Fill(nDigis);

    double maxAmplitude = 0.;

    if (doSim_) {
      const edm::PSimHitContainer simHits = theSimHitMap->hits(detId);
      if (nDigis == 1 && simHits.size() == 1) {
        plotResolution(simHits[0], digiItr->getStrip(), layer, chamberType);
      }
    }

    for (; digiItr != last; ++digiItr) {
      // average up the pedestals
      std::vector<int> adcCounts = digiItr->getADCCounts();
      thePedestalSum += adcCounts[0];
      thePedestalSum += adcCounts[1];
      thePedestalCount += 2;
      float pedestal = thePedestalSum / thePedestalCount;
      if (adcCounts[4] - pedestal > maxAmplitude) {
        maxAmplitude = adcCounts[4] - pedestal;
      }

      // if we have enough pedestal statistics
      if (thePedestalCount > 100) {
        fillPedestalPlots(*digiItr);

        // see if it's big enough to count as "signal"
        if (adcCounts[5] > (thePedestalSum / thePedestalCount + 100)) {
          fillSignalPlots(*digiItr);
        }
      }
    }
  }  // loop over digis

  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}

void CSCStripDigiValidation::fillPedestalPlots(const CSCStripDigi &digi) {
  std::vector<int> adcCounts = digi.getADCCounts();
  thePedestalPlot->Fill(adcCounts[0]);
  thePedestalPlot->Fill(adcCounts[1]);
}

void CSCStripDigiValidation::fillSignalPlots(const CSCStripDigi &digi) {
  std::vector<int> adcCounts = digi.getADCCounts();
  float pedestal = thePedestalSum / thePedestalCount;
  theAmplitudePlot->Fill(adcCounts[4] - pedestal);
  theRatio4to5Plot->Fill((adcCounts[3] - pedestal) / (adcCounts[4] - pedestal));
  theRatio6to5Plot->Fill((adcCounts[5] - pedestal) / (adcCounts[4] - pedestal));
}

void CSCStripDigiValidation::plotResolution(const PSimHit &hit, int strip, const CSCLayer *layer, int chamberType) {
  double hitX = hit.localPosition().x();
  double hitY = hit.localPosition().y();
  double digiX = layer->geometry()->xOfStrip(strip, hitY);
  theResolutionPlots[chamberType - 1]->Fill(digiX - hitX);
}
