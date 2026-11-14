#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "Validation/RecoMET/plugins/METTesterPostProcessorHarvesting.h"

#include <algorithm>

METTesterPostProcessorHarvesting::METTesterPostProcessorHarvesting(const edm::ParameterSet &iConfig) {
  runDir = iConfig.getUntrackedParameter<std::string>("runDir");
  if (!runDir.empty() and runDir.back() != '/')
    runDir += '/';
  inputMETLabelRECO_ = iConfig.getParameter<edm::InputTag>("METTypeRECO");
  inputMETLabelMiniAOD_ = iConfig.getParameter<edm::InputTag>("METTypeMiniAOD");
}

METTesterPostProcessorHarvesting::~METTesterPostProcessorHarvesting() {}

// ------------ method called right after a run ends ------------
void METTesterPostProcessorHarvesting::dqmEndJob(DQMStore::IBooker &ibook_, DQMStore::IGetter &iget_) {
  iget_.setCurrentFolder(runDir);
  const std::vector<std::string> met_dirs = iget_.getSubdirs();

  const std::string rundir_reco = runDir + inputMETLabelRECO_.label();
  const std::string rundir_miniaod = runDir + inputMETLabelMiniAOD_.label();
  const bool found_reco_dir = std::find(met_dirs.begin(), met_dirs.end(), rundir_reco) != met_dirs.end();
  const bool found_miniaod_dir = std::find(met_dirs.begin(), met_dirs.end(), rundir_miniaod) != met_dirs.end();
  if (!found_reco_dir or !found_miniaod_dir)
    return;

  ibook_.setCurrentFolder(runDir + "MiniAOD_over_RECO");
  for (const auto &[name, outName] : mHistoNames) {
    MonitorElement *monReco = iget_.get(rundir_reco + "/" + name);
    MonitorElement *monMiniAOD = iget_.get(rundir_miniaod + "/" + name);
    if (!monReco or monReco->kind() != MonitorElement::Kind::TH1F or !monMiniAOD or
        monMiniAOD->kind() != MonitorElement::Kind::TH1F) {
      edm::LogInfo("METTesterPostProcessorHarvesting")
          << "Histogram " << name << " missing in " << rundir_reco << " or " << rundir_miniaod << "; skipping.";
      continue;
    }
    if (monReco->getNbinsX() != monMiniAOD->getNbinsX()) {
      edm::LogWarning("METTesterPostProcessorHarvesting")
          << "Histogram " << name << " has different binning in RECO and MiniAOD; skipping.";
      continue;
    }

    MonitorElement *monMiniAOD_over_RECO = ibook_.book1D(outName, monReco->getTH1F());
    for (int i = 0; i <= (monMiniAOD_over_RECO->getNbinsX() + 1); i++) {
      if (monReco->getBinContent(i) != 0) {
        monMiniAOD_over_RECO->setBinContent(i, monMiniAOD->getBinContent(i) / monReco->getBinContent(i));
      } else if (monMiniAOD->getBinContent(i) != 0) {
        monMiniAOD_over_RECO->setBinContent(i, -0.5);
      } else {
        monMiniAOD_over_RECO->setBinContent(i, 0.);
      }
    }
  }
}

void METTesterPostProcessorHarvesting::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("runDir", "JetMET/METValidation/");
  desc.add<edm::InputTag>("METTypeRECO", edm::InputTag("PfMetT1"));
  desc.add<edm::InputTag>("METTypeMiniAOD", edm::InputTag("slimmedMETs"));
  descriptions.addWithDefaultLabel(desc);
}
