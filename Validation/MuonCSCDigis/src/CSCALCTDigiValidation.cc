#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCALCTDigiValidation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

CSCALCTDigiValidation::CSCALCTDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps),
      theNDigisPerChamberPlots(),
      chambers_(ps.getParameter<std::vector<std::string>>("chambers")),
      // variables
      alctVars_(ps.getParameter<std::vector<std::string>>("alctVars")),
      // binning
      alctNBin_(ps.getParameter<std::vector<unsigned>>("alctNBin")),
      alctMinBin_(ps.getParameter<std::vector<double>>("alctMinBin")),
      alctMaxBin_(ps.getParameter<std::vector<double>>("alctMaxBin")) {
  const auto &pset = ps.getParameterSet("cscALCT");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  alcts_Token_ = iC.consumes<CSCALCTDigiCollection>(inputTag_);
}

CSCALCTDigiValidation::~CSCALCTDigiValidation() {}

void CSCALCTDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/ALCT/Occupancy/");

  theNDigisPerEventPlot = iBooker.book1D(
      "CSCALCTDigisPerEvent", "ALCT trigger primitives per event; Number of ALCTs; Entries", 100, 0, 100);
  for (int i = 1; i <= 10; ++i) {
    const std::string t2("CSCALCTDigisPerChamber_" + CSCDetId::chamberName(i));
    theNDigisPerChamberPlots[i - 1] = iBooker.book1D(
        t2, "Number of ALCTs per chamber " + CSCDetId::chamberName(i) + ";Number of ALCTs per chamber;Entries", 4, 0, 4);
  }

  // chamber type
  for (unsigned iType = 0; iType < chambers_.size(); iType++) {
    // consider CSC+ and CSC- separately
    for (unsigned iEndcap = 0; iEndcap < 2; iEndcap++) {
      const std::string eSign(iEndcap == 0 ? "+" : "-");
      // alct variable
      for (unsigned iVar = 0; iVar < alctVars_.size(); iVar++) {
        const std::string key("alct_" + alctVars_[iVar]);
        const std::string histName(key + "_" + chambers_[iType] + eSign);
        const std::string histTitle(chambers_[iType] + eSign + " ALCT " + alctVars_[iVar]);
        const unsigned iTypeCorrected(iEndcap == 0 ? iType : iType + chambers_.size());
        chamberHistos[iTypeCorrected][key] =
            iBooker.book1D(histName, histTitle, alctNBin_[iVar], alctMinBin_[iVar], alctMaxBin_[iVar]);
        chamberHistos[iTypeCorrected][key]->getTH1()->SetMinimum(0);
      }
    }
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
    const CSCDetId &detId((*j).first);
    int chamberType = detId.iChamberType();
    int nDigis = endDigi - beginDigi;
    nDigisPerEvent += nDigis;
    theNDigisPerChamberPlots[chamberType - 1]->Fill(nDigis);

    auto range = alcts->get((*j).first);
    // 1=forward (+Z); 2=backward (-Z)
    const unsigned typeCorrected(detId.endcap() == 1 ? chamberType - 2 : chamberType - 2 + chambers_.size());
    for (auto alct = range.first; alct != range.second; alct++) {
      if (alct->isValid()) {
        chamberHistos[typeCorrected]["alct_quality"]->Fill(alct->getQuality());
        chamberHistos[typeCorrected]["alct_wiregroup"]->Fill(alct->getKeyWG());
        chamberHistos[typeCorrected]["alct_bx"]->Fill(alct->getBX());
      }
    }
  }
  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}
