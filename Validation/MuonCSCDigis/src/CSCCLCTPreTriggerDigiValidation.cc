#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCCLCTPreTriggerDigiValidation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

CSCCLCTPreTriggerDigiValidation::CSCCLCTPreTriggerDigiValidation(const edm::ParameterSet &ps,
                                                                 edm::ConsumesCollector &&iC)
    : CSCBaseValidation(ps),
      chambers_(ps.getParameter<std::vector<std::string>>("chambers")),
      // variables
      preclctVars_(ps.getParameter<std::vector<std::string>>("preclctVars")),
      // binning
      preclctNBin_(ps.getParameter<std::vector<unsigned>>("preclctNBin")),
      preclctMinBin_(ps.getParameter<std::vector<double>>("preclctMinBin")),
      preclctMaxBin_(ps.getParameter<std::vector<double>>("preclctMaxBin")) {
  const auto &pset = ps.getParameterSet("cscCLCTPreTrigger");
  inputTag_ = pset.getParameter<edm::InputTag>("inputTag");
  preclcts_Token_ = iC.consumes<CSCCLCTPreTriggerDigiCollection>(inputTag_);
}

CSCCLCTPreTriggerDigiValidation::~CSCCLCTPreTriggerDigiValidation() {}

void CSCCLCTPreTriggerDigiValidation::bookHistograms(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/PreCLCT/Occupancy/");

  // chamber type
  for (unsigned iType = 0; iType < chambers_.size(); iType++) {
    // consider CSC+ and CSC- separately
    for (unsigned iEndcap = 0; iEndcap < 2; iEndcap++) {
      const std::string eSign(iEndcap == 0 ? "+" : "-");
      // preclct variable
      for (unsigned iVar = 0; iVar < preclctVars_.size(); iVar++) {
        const std::string key("preclct_" + preclctVars_[iVar]);
        const std::string histName(key + "_" + chambers_[iType] + eSign);
        const std::string histTitle(chambers_[iType] + eSign + " CLCTPreTrigger " + preclctVars_[iVar]);
        const unsigned iTypeCorrected(iEndcap == 0 ? iType : iType + chambers_.size());
        chamberHistos[iTypeCorrected][key] =
            iBooker.book1D(histName, histTitle, preclctNBin_[iVar], preclctMinBin_[iVar], preclctMaxBin_[iVar]);
        chamberHistos[iTypeCorrected][key]->getTH1()->SetMinimum(0);
      }
    }
  }
}

void CSCCLCTPreTriggerDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &) {
  edm::Handle<CSCCLCTPreTriggerDigiCollection> preclcts;
  e.getByToken(preclcts_Token_, preclcts);
  if (!preclcts.isValid()) {
    edm::LogError("CSCCLCTPreTriggerDigiValidation") << "Cannot get CLCTPreTriggers by label " << inputTag_.encode();
  }

  for (auto j = preclcts->begin(); j != preclcts->end(); j++) {
    const CSCDetId &detId((*j).first);
    int chamberType = detId.iChamberType();

    auto range = preclcts->get((*j).first);
    // 1=forward (+Z); 2=backward (-Z)
    const unsigned typeCorrected(detId.endcap() == 1 ? chamberType - 2 : chamberType - 2 + chambers_.size());
    for (auto preclct = range.first; preclct != range.second; preclct++) {
      if (preclct->isValid()) {
        chamberHistos[typeCorrected]["preclct_cfeb"]->Fill(preclct->getCFEB());
        chamberHistos[typeCorrected]["preclct_halfstrip"]->Fill(preclct->getKeyStrip());
        chamberHistos[typeCorrected]["preclct_bx"]->Fill(preclct->getBX());
      }
    }
  }
}
