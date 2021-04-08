#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCWireDigiValidation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

CSCWireDigiValidation::CSCWireDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps), theTimeBinPlots(), theNDigisPerLayerPlots() {
  const auto &pset = ps.getParameterSet("cscWireDigi");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  wires_Token_ = iC.consumes<CSCWireDigiCollection>(inputTag_);
}

CSCWireDigiValidation::~CSCWireDigiValidation() {}

void CSCWireDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  theNDigisPerEventPlot =
      iBooker.book1D("CSCWireDigisPerEvent", "CSC Wire Digis per event;CSC Wire Digis per event;Entries", 100, 0, 100);
  for (int i = 1; i <= 10; ++i) {
    const std::string t1("CSCWireDigiTime_" + CSCDetId::chamberName(i));
    const std::string t2("CSCWireDigisPerLayer_" + CSCDetId::chamberName(i));
    const std::string t3("CSCWireDigiResolution_" + CSCDetId::chamberName(i));
    theTimeBinPlots[i - 1] =
        iBooker.book1D(t1, "Wire Time Bin " + CSCDetId::chamberName(i) + ";Wire Time Bin; Entries", 16, 0, 16);
    theNDigisPerLayerPlots[i - 1] = iBooker.book1D(
        t2, "Number of Wire Digis " + CSCDetId::chamberName(i) + ";Number of Wire Digis; Entries", 100, 0, 20);
    theResolutionPlots[i - 1] = iBooker.book1D(
        t3,
        "Wire Y Position Resolution " + CSCDetId::chamberName(i) + ";Wire Y Position Resolution; Entries",
        100,
        -10,
        10);
  }
}

void CSCWireDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &) {
  edm::Handle<CSCWireDigiCollection> wires;

  e.getByToken(wires_Token_, wires);

  if (!wires.isValid()) {
    edm::LogError("CSCWireDigiValidation") << "Cannot get wires by label " << inputTag_.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (auto j = wires->begin(); j != wires->end(); j++) {
    auto beginDigi = (*j).second.first;
    auto endDigi = (*j).second.second;
    int detId = (*j).first.rawId();

    const CSCLayer *layer = findLayer(detId);
    int chamberType = layer->chamber()->specs()->chamberType();
    int nDigis = endDigi - beginDigi;
    nDigisPerEvent += nDigis;
    theNDigisPerLayerPlots[chamberType - 1]->Fill(nDigis);

    for (auto digiItr = beginDigi; digiItr != endDigi; ++digiItr) {
      theTimeBinPlots[chamberType - 1]->Fill(digiItr->getTimeBin());
    }

    if (doSim_) {
      const edm::PSimHitContainer simHits = theSimHitMap->hits(detId);
      if (nDigis == 1 && simHits.size() == 1) {
        plotResolution(simHits[0], *beginDigi, layer, chamberType);
      }
    }
  }

  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}

void CSCWireDigiValidation::plotResolution(const PSimHit &hit,
                                           const CSCWireDigi &digi,
                                           const CSCLayer *layer,
                                           int chamberType) {
  double hitX = hit.localPosition().x();
  double hitY = hit.localPosition().y();
  double digiY = layer->geometry()->yOfWireGroup(digi.getWireGroup(), hitX);
  theResolutionPlots[chamberType - 1]->Fill(digiY - hitY);
}
