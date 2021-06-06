#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCComparatorDigiValidation.h"

CSCComparatorDigiValidation::CSCComparatorDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps), theTimeBinPlots(), theNDigisPerLayerPlots(), theStripDigiPlots(), the3StripPlots() {
  const auto &comps = ps.getParameterSet("cscComparatorDigi");
  inputTagComp_ = comps.getParameter<edm::InputTag>("inputTag");
  comparators_Token_ = iC.consumes<CSCComparatorDigiCollection>(inputTagComp_);

  const auto &strips = ps.getParameterSet("cscStripDigi");
  inputTagStrip_ = strips.getParameter<edm::InputTag>("inputTag");
  strips_Token_ = iC.consumes<CSCStripDigiCollection>(inputTagStrip_);
}

CSCComparatorDigiValidation::~CSCComparatorDigiValidation() {}

void CSCComparatorDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  theNDigisPerEventPlot = iBooker.book1D("CSCComparatorDigisPerEvent",
                                         "CSC Comparator Digis per event;CSC Comparator Digis per event;Entries",
                                         100,
                                         0,
                                         100);
  // 10 chamber types, if you consider ME1/a and ME1/b separate
  for (int i = 1; i <= 10; ++i) {
    const std::string t1("CSCComparatorDigiTime_" + CSCDetId::chamberName(i));
    const std::string t2("CSCComparatorDigisPerLayer_" + CSCDetId::chamberName(i));
    const std::string t3("CSCComparatorStripAmplitude_" + CSCDetId::chamberName(i));
    const std::string t4("CSCComparator3StripAmplitude_" + CSCDetId::chamberName(i));
    theTimeBinPlots[i - 1] = iBooker.book1D(
        t1, "Comparator Time Bin " + CSCDetId::chamberName(i) + " ;Comparator Time Bin; Entries", 16, 0, 16);
    theNDigisPerLayerPlots[i - 1] = iBooker.book1D(
        t2,
        "Number of Comparator Digis " + CSCDetId::chamberName(i) + " ;Number of Comparator Digis; Entries",
        100,
        0,
        20);
    theStripDigiPlots[i - 1] = iBooker.book1D(
        t3, "Comparator Amplitude " + CSCDetId::chamberName(i) + " ;Comparator Amplitude; Entries", 100, 0, 1000);
    the3StripPlots[i - 1] = iBooker.book1D(
        t4,
        "Comparator-triplet Amplitude " + CSCDetId::chamberName(i) + " ;Comparator-triplet Amplitude; Entries",
        100,
        0,
        1000);
  }
}

void CSCComparatorDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &) {
  edm::Handle<CSCComparatorDigiCollection> comparators;
  edm::Handle<CSCStripDigiCollection> stripDigis;

  e.getByToken(comparators_Token_, comparators);
  if (!comparators.isValid()) {
    edm::LogError("CSCComparatorDigiValidation") << "Cannot get comparators by label " << inputTagComp_.encode();
  }
  e.getByToken(strips_Token_, stripDigis);
  if (!stripDigis.isValid()) {
    edm::LogError("CSCComparatorDigiValidation") << "Cannot get strips by label " << inputTagComp_.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (auto j = comparators->begin(); j != comparators->end(); j++) {
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;

    CSCDetId detId((*j).first);
    const CSCLayer *layer = findLayer(detId.rawId());
    int chamberType = layer->chamber()->specs()->chamberType();

    theNDigisPerLayerPlots[chamberType - 1]->Fill(last - digiItr);

    for (auto stripRange = stripDigis->get(detId); digiItr != last; ++digiItr) {
      ++nDigisPerEvent;
      theTimeBinPlots[chamberType - 1]->Fill(digiItr->getTimeBin());

      int strip = digiItr->getStrip();
      for (auto stripItr = stripRange.first; stripItr != stripRange.second; ++stripItr) {
        if (stripItr->getStrip() == strip) {
          std::vector<int> adc = stripItr->getADCCounts();
          float pedc = 0.5 * (adc[0] + adc[1]);
          float amp = adc[4] - pedc;
          theStripDigiPlots[chamberType - 1]->Fill(amp);
          // check neighbors
          if (stripItr != stripRange.first && stripItr != stripRange.second - 1) {
            std::vector<int> adcl = (stripItr - 1)->getADCCounts();
            std::vector<int> adcr = (stripItr + 1)->getADCCounts();
            float pedl = 0.5 * (adcl[0] + adcl[1]);
            float pedr = 0.5 * (adcr[0] + adcr[1]);
            float three = adcl[4] - pedl + adcr[4] - pedr + amp;
            the3StripPlots[chamberType - 1]->Fill(three);
          }
        }
      }
    }
  }
  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}
