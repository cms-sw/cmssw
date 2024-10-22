#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCCorrelatedLCTDigiValidation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

CSCCorrelatedLCTDigiValidation::CSCCorrelatedLCTDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps),
      theNDigisPerChamberPlots(),
      chambers_(ps.getParameter<std::vector<std::string>>("chambers")),
      chambersRun3_(ps.getParameter<std::vector<unsigned>>("chambersRun3")),
      // variables
      lctVars_(ps.getParameter<std::vector<std::string>>("lctVars")),
      // binning
      lctNBin_(ps.getParameter<std::vector<unsigned>>("lctNBin")),
      lctMinBin_(ps.getParameter<std::vector<double>>("lctMinBin")),
      lctMaxBin_(ps.getParameter<std::vector<double>>("lctMaxBin")),
      isRun3_(ps.getParameter<bool>("isRun3")) {
  const auto &pset = ps.getParameterSet("cscMPLCT");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  lcts_Token_ = iC.consumes<CSCCorrelatedLCTDigiCollection>(inputTag_);
}

CSCCorrelatedLCTDigiValidation::~CSCCorrelatedLCTDigiValidation() {}

void CSCCorrelatedLCTDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/LCT/Occupancy/");

  theNDigisPerEventPlot =
      iBooker.book1D("CSCCorrelatedLCTDigisPerEvent",
                     "CorrelatedLCT trigger primitives per event; Number of CorrelatedLCTs; Entries",
                     100,
                     0,
                     100);
  for (int i = 1; i <= 10; ++i) {
    const std::string t2("CSCCorrelatedLCTDigisPerChamber_" + CSCDetId::chamberName(i));
    theNDigisPerChamberPlots[i - 1] =
        iBooker.book1D(t2,
                       "Number of CorrelatedLCTs per chamber " + CSCDetId::chamberName(i) +
                           ";Number of CorrelatedLCTs per chamber;Entries",
                       4,
                       0,
                       4);
  }

  // do not analyze Run-3 properties in Run-1 and Run-2 eras
  if (!isRun3_) {
    lctVars_.resize(7);
  }

  // chamber type
  for (unsigned iType = 0; iType < chambers_.size(); iType++) {
    // consider CSC+ and CSC- separately
    for (unsigned iEndcap = 0; iEndcap < 2; iEndcap++) {
      const std::string eSign(iEndcap == 0 ? "+" : "-");
      // lct variable
      for (unsigned iVar = 0; iVar < lctVars_.size(); iVar++) {
        if (std::find(chambersRun3_.begin(), chambersRun3_.end(), iType) == chambersRun3_.end()) {
          if (iVar > 6)
            continue;
        }
        const std::string key("lct_" + lctVars_[iVar]);
        const std::string histName(key + "_" + chambers_[iType] + eSign);
        const std::string histTitle(chambers_[iType] + eSign + " LCT " + lctVars_[iVar]);
        const unsigned iTypeCorrected(iEndcap == 0 ? iType : iType + chambers_.size());
        chamberHistos[iTypeCorrected][key] =
            iBooker.book1D(histName, histTitle, lctNBin_[iVar], lctMinBin_[iVar], lctMaxBin_[iVar]);
        chamberHistos[iTypeCorrected][key]->getTH1()->SetMinimum(0);
        // set bin labels for the "type" plot. very useful in ME1/1 and ME2/1
        // when the GEM-CSC ILTs will be running
        if (lctVars_[iVar] == "type") {
          chamberHistos[iTypeCorrected][key]->setBinLabel(1, "CLCTALCT", 1);
          chamberHistos[iTypeCorrected][key]->setBinLabel(2, "ALCTCLCT", 1);
          chamberHistos[iTypeCorrected][key]->setBinLabel(3, "ALCTCLCTGEM", 1);
          chamberHistos[iTypeCorrected][key]->setBinLabel(4, "ALCTCLCT2GEM", 1);
          chamberHistos[iTypeCorrected][key]->setBinLabel(5, "ALCT2GEM", 1);
          chamberHistos[iTypeCorrected][key]->setBinLabel(6, "CLCT2GEM", 1);
          chamberHistos[iTypeCorrected][key]->setBinLabel(7, "CLCTONLY", 1);
          chamberHistos[iTypeCorrected][key]->setBinLabel(8, "ALCTONLY", 1);
        }
      }
    }
  }
}

void CSCCorrelatedLCTDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &) {
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts;
  e.getByToken(lcts_Token_, lcts);
  if (!lcts.isValid()) {
    edm::LogError("CSCDigiDump") << "Cannot get CorrelatedLCTs by label " << inputTag_.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (auto j = lcts->begin(); j != lcts->end(); j++) {
    auto beginDigi = (*j).second.first;
    auto endDigi = (*j).second.second;
    const CSCDetId &detId((*j).first);
    int chamberType = detId.iChamberType();
    int nDigis = endDigi - beginDigi;
    nDigisPerEvent += nDigis;
    theNDigisPerChamberPlots[chamberType - 1]->Fill(nDigis);

    auto range = lcts->get((*j).first);
    // 1=forward (+Z); 2=backward (-Z)
    const unsigned typeCorrected(detId.endcap() == 1 ? chamberType - 2 : chamberType - 2 + chambers_.size());
    for (auto lct = range.first; lct != range.second; lct++) {
      if (lct->isValid()) {
        chamberHistos[typeCorrected]["lct_pattern"]->Fill(lct->getPattern());
        chamberHistos[typeCorrected]["lct_quality"]->Fill(lct->getQuality());
        chamberHistos[typeCorrected]["lct_wiregroup"]->Fill(lct->getKeyWG());
        chamberHistos[typeCorrected]["lct_halfstrip"]->Fill(lct->getStrip());
        chamberHistos[typeCorrected]["lct_bend"]->Fill(lct->getBend());
        chamberHistos[typeCorrected]["lct_bx"]->Fill(lct->getBX());
        chamberHistos[typeCorrected]["lct_type"]->Fill(lct->getType());
        if (isRun3_) {
          // ignore these fields for chambers that do not enable the Run-3 algorithm
          if (std::find(chambersRun3_.begin(), chambersRun3_.end(), chamberType - 2) == chambersRun3_.end())
            continue;
          chamberHistos[typeCorrected]["lct_run3pattern"]->Fill(lct->getRun3Pattern());
          chamberHistos[typeCorrected]["lct_slope"]->Fill(lct->getSlope());
          chamberHistos[typeCorrected]["lct_quartstrip"]->Fill(lct->getStrip(4));
          chamberHistos[typeCorrected]["lct_eighthstrip"]->Fill(lct->getStrip(8));
        }
      }
    }
  }
  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}
