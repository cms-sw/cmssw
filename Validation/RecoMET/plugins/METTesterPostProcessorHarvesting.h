#ifndef METTESTERPOSTPROCESSORHARVESTING_H
#define METTESTERPOSTPROCESSORHARVESTING_H

// author: Matthias Weber, Feb 2015

// system include files
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//
// class declaration
//

class METTesterPostProcessorHarvesting : public DQMEDHarvester {
public:
  explicit METTesterPostProcessorHarvesting(const edm::ParameterSet &);
  ~METTesterPostProcessorHarvesting() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  std::string runDir;
  edm::InputTag inputMETLabelRECO_;
  edm::InputTag inputMETLabelMiniAOD_;

  // histograms compared between RECO and MiniAOD: (input name, output name)
  const std::vector<std::pair<std::string, std::string>> mHistoNames = {
      {"MET", "MET_MiniAOD_over_RECO"},
      {"METPhi", "METPhi_MiniAOD_over_RECO"},
      {"SumET", "SumET_MiniAOD_over_RECO"},
      {"METDiff_GenMETTrue", "METDiff_GenMETTrue_MiniAOD_over_RECO"},
      {"METDeltaPhi_GenMETTrue", "METDeltaPhi_GenMETTrue_MiniAOD_over_RECO"},
      {"photonEtFraction", "photonEtFraction_MiniAOD_over_RECO"},
      {"neutralHadronEtFraction", "neutralHadronEtFraction_MiniAOD_over_RECO"},
      {"chargedHadronEtFraction", "chargedHadronEtFraction_MiniAOD_over_RECO"},
      {"HFHadronEtFraction", "HFHadronEtFraction_MiniAOD_over_RECO"},
      {"HFEMEtFraction", "HFEMEtEtFraction_MiniAOD_over_RECO"},
  };
};

#endif
