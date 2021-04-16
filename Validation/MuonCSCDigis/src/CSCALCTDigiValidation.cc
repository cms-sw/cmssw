#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCALCTDigiValidation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

CSCALCTDigiValidation::CSCALCTDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps), theTimeBinPlots(), theNDigisPerChamberPlots() {
  const auto &pset = ps.getParameterSet("cscALCT");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  alcts_Token_ = iC.consumes<CSCALCTDigiCollection>(inputTag_);
}

CSCALCTDigiValidation::~CSCALCTDigiValidation() {}

void CSCALCTDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  theNDigisPerEventPlot = iBooker.book1D(
      "CSCALCTDigisPerEvent", "ALCT trigger primitives per event; Number of ALCTs; Entries", 100, 0, 100);
  for (int i = 1; i <= 10; ++i) {
    const std::string t1("CSCALCTDigiTime_" + CSCDetId::chamberName(i));
    const std::string t2("CSCALCTDigisPerChamber_" + CSCDetId::chamberName(i));
    theTimeBinPlots[i - 1] = iBooker.book1D(t1, "ALCT BX " + CSCDetId::chamberName(i) + ";ALCT BX; Entries", 16, 0, 16);
    theNDigisPerChamberPlots[i - 1] = iBooker.book1D(
        t2, "Number of ALCTs per chamber " + CSCDetId::chamberName(i) + ";Number of ALCTs per chamber;Entries", 4, 0, 4);
  }
}

void CSCALCTDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &) {
  edm::Handle<CSCALCTDigiCollection> alcts;
  e.getByToken(alcts_Token_, alcts);
  if (!alcts.isValid()) {
    edm::LogError("CSCALCTDigiValidation") << "Cannot get ALCTs by label " << inputTag_.encode();
  }
  unsigned nDigisPerEvent = 0;

  for (auto j = alcts->begin(); j != alcts->end(); j++) {
    auto beginDigi = (*j).second.first;
    auto endDigi = (*j).second.second;
    CSCDetId detId((*j).first.rawId());
    int chamberType = detId.iChamberType();
    int nDigis = endDigi - beginDigi;
    nDigisPerEvent += nDigis;
    theNDigisPerChamberPlots[chamberType - 1]->Fill(nDigis);

    for (auto digiItr = beginDigi; digiItr != endDigi; ++digiItr) {
      theTimeBinPlots[chamberType - 1]->Fill(digiItr->getBX());
    }
  }
  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}
