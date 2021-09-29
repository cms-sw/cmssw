#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCCLCTDigiValidation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

CSCCLCTDigiValidation::CSCCLCTDigiValidation(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps),
      theNDigisPerChamberPlots(),
      chambers_(ps.getParameter<std::vector<std::string>>("chambers")),
      chambersRun3_(ps.getParameter<std::vector<unsigned>>("chambersRun3")),
      // variables
      clctVars_(ps.getParameter<std::vector<std::string>>("clctVars")),
      // binning
      clctNBin_(ps.getParameter<std::vector<unsigned>>("clctNBin")),
      clctMinBin_(ps.getParameter<std::vector<double>>("clctMinBin")),
      clctMaxBin_(ps.getParameter<std::vector<double>>("clctMaxBin")),
      isRun3_(ps.getParameter<bool>("isRun3")) {
  const auto &pset = ps.getParameterSet("cscCLCT");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  clcts_Token_ = iC.consumes<CSCCLCTDigiCollection>(inputTag_);
}

CSCCLCTDigiValidation::~CSCCLCTDigiValidation() {}

void CSCCLCTDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/CLCT/Occupancy/");

  theNDigisPerEventPlot = iBooker.book1D(
      "CSCCLCTDigisPerEvent", "CLCT trigger primitives per event; Number of CLCTs; Entries", 100, 0, 100);
  for (int i = 1; i <= 10; ++i) {
    const std::string t2("CSCCLCTDigisPerChamber_" + CSCDetId::chamberName(i));
    theNDigisPerChamberPlots[i - 1] = iBooker.book1D(
        t2, "Number of CLCTs per chamber " + CSCDetId::chamberName(i) + ";Number of CLCTs per chamber;Entries", 4, 0, 4);
  }

  // do not analyze Run-3 properties in Run-1 and Run-2 eras
  if (!isRun3_) {
    clctVars_.resize(5);
  }

  // chamber type
  for (unsigned iType = 0; iType < chambers_.size(); iType++) {
    // consider CSC+ and CSC- separately
    for (unsigned iEndcap = 0; iEndcap < 2; iEndcap++) {
      const std::string eSign(iEndcap == 0 ? "+" : "-");
      // clct variable
      for (unsigned iVar = 0; iVar < clctVars_.size(); iVar++) {
        if (std::find(chambersRun3_.begin(), chambersRun3_.end(), iType) == chambersRun3_.end()) {
          if (iVar > 4)
            continue;
        }
        const std::string key("clct_" + clctVars_[iVar]);
        const std::string histName(key + "_" + chambers_[iType] + eSign);
        const std::string histTitle(chambers_[iType] + eSign + " CLCT " + clctVars_[iVar]);
        const unsigned iTypeCorrected(iEndcap == 0 ? iType : iType + chambers_.size());
        chamberHistos[iTypeCorrected][key] =
            iBooker.book1D(histName, histTitle, clctNBin_[iVar], clctMinBin_[iVar], clctMaxBin_[iVar]);
        chamberHistos[iTypeCorrected][key]->getTH1()->SetMinimum(0);
      }
    }
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
    const CSCDetId &detId((*j).first);
    int chamberType = detId.iChamberType();
    int nDigis = endDigi - beginDigi;
    nDigisPerEvent += nDigis;
    theNDigisPerChamberPlots[chamberType - 1]->Fill(nDigis);

    auto range = clcts->get((*j).first);
    // 1=forward (+Z); 2=backward (-Z)
    const unsigned typeCorrected(detId.endcap() == 1 ? chamberType - 2 : chamberType - 2 + chambers_.size());
    for (auto clct = range.first; clct != range.second; clct++) {
      if (clct->isValid()) {
        chamberHistos[typeCorrected]["clct_pattern"]->Fill(clct->getPattern());
        chamberHistos[typeCorrected]["clct_quality"]->Fill(clct->getQuality());
        chamberHistos[typeCorrected]["clct_halfstrip"]->Fill(clct->getKeyStrip());
        chamberHistos[typeCorrected]["clct_bend"]->Fill(clct->getBend());
        chamberHistos[typeCorrected]["clct_bx"]->Fill(clct->getBX());
        if (isRun3_) {
          // ignore these fields for chambers that do not enable the Run-3 algorithm
          if (std::find(chambersRun3_.begin(), chambersRun3_.end(), chamberType - 2) == chambersRun3_.end())
            continue;
          chamberHistos[typeCorrected]["clct_run3pattern"]->Fill(clct->getRun3Pattern());
          chamberHistos[typeCorrected]["clct_quartstrip"]->Fill(clct->getKeyStrip(4));
          chamberHistos[typeCorrected]["clct_eighthstrip"]->Fill(clct->getKeyStrip(8));
          chamberHistos[typeCorrected]["clct_slope"]->Fill(clct->getSlope());
          chamberHistos[typeCorrected]["clct_compcode"]->Fill(clct->getCompCode());
        }
      }
    }
  }
  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}
