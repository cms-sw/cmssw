//
//
/**
    @file PatMuonHitFitTranslator.cc

    @brief Specialization of template class LeptonTranslatorBase in the
    package HitFit for pat::Muon.

    @author Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Created
    Sat Jun 27 17:49:15 2009 UTC

    @version $Id: PatMuonHitFitTranslator.cc,v 1.8 2010/08/06 22:03:03 haryo Exp $
 */

#include "TopQuarkAnalysis/TopHitFit/interface/LeptonTranslatorBase.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

namespace hitfit {

  template <>
  LeptonTranslatorBase<pat::Muon>::LeptonTranslatorBase() {
    std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
    std::string resolution_filename =
        CMSSW_BASE + std::string("/src/TopQuarkAnalysis/PatHitFit/data/exampleMuonResolution.txt");
    resolution_ = EtaDepResolution(resolution_filename);

  }  // LeptonTranslatorBase<pat::Muon>::LeptonTranslatorBase()

  template <>
  LeptonTranslatorBase<pat::Muon>::LeptonTranslatorBase(const std::string& ifile) {
    std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
    std::string resolution_filename;

    if (ifile.empty()) {
      resolution_filename = CMSSW_BASE + std::string("/src/TopQuarkAnalysis/PatHitFit/data/exampleMuonResolution.txt");
    } else {
      resolution_filename = ifile;
    }

    resolution_ = EtaDepResolution(resolution_filename);

  }  // LeptonTranslatorBase<pat::Muon>::LeptonTranslatorBase(const std::string& s)

  template <>
  LeptonTranslatorBase<pat::Muon>::~LeptonTranslatorBase() {
  }  // LeptonTranslatorBase<pat::Muon>::~LeptonTranslatorBase()

  template <>
  Lepjets_Event_Lep LeptonTranslatorBase<pat::Muon>::operator()(const pat::Muon& lepton,
                                                                int type /*= hitfit::lepton_label */,
                                                                bool useObjEmbRes /* = false */) {
    Fourvec p(lepton.px(), lepton.py(), lepton.pz(), lepton.energy());

    double muon_eta = lepton.eta();
    Vector_Resolution muon_resolution = resolution_.GetResolution(muon_eta);

    Lepjets_Event_Lep muon(p, muon_label, muon_resolution);
    return muon;

  }  // Lepjets_Event_Lep LeptonTranslatorBase<pat::Muon>::operator()

  template <>
  const EtaDepResolution& LeptonTranslatorBase<pat::Muon>::resolution() const {
    return resolution_;
  }

  template <>
  bool LeptonTranslatorBase<pat::Muon>::CheckEta(const pat::Muon& lepton) const {
    return resolution_.CheckEta(lepton.eta());
  }

}  // namespace hitfit
