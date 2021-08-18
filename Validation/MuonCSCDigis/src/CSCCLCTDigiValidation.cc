#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCCLCTDigiValidation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

CSCCLCTDigiValidation::CSCCLCTDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps), theTimeBinPlots(), theNDigisPerChamberPlots() {
  const auto &pset = ps.getParameterSet("cscCLCT");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  clcts_Token_ = iC.consumes<CSCCLCTDigiCollection>(inputTag_);
}

CSCCLCTDigiValidation::~CSCCLCTDigiValidation() {}

void CSCCLCTDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  theNDigisPerEventPlot = iBooker.book1D(
      "CSCCLCTDigisPerEvent", "CLCT trigger primitives per event; Number of CLCTs; Entries", 100, 0, 100);
  for (int i = 1; i <= 10; ++i) {
    const std::string t1("CSCCLCTDigiTime_" + CSCDetId::chamberName(i));
    const std::string t2("CSCCLCTDigisPerChamber_" + CSCDetId::chamberName(i));
    theTimeBinPlots[i - 1] = iBooker.book1D(t1, "CLCT BX " + CSCDetId::chamberName(i) + ";CLCT BX; Entries", 16, 0, 16);
    theNDigisPerChamberPlots[i - 1] = iBooker.book1D(
        t2, "Number of CLCTs per chamber " + CSCDetId::chamberName(i) + ";Number of CLCTs per chamber;Entries", 4, 0, 4);
  }
}

void CSCCLCTDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &) {
  edm::Handle<CSCCLCTDigiCollection> clcts;
  e.getByToken(clcts_Token_, clcts);
  if (!clcts.isValid()) {
    edm::LogError("CSCDigiDump") << "Cannot get CLCTs by label " << inputTag_.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (auto j = clcts->begin(); j != clcts->end(); j++) {
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
