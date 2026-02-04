#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "Validation/RecoMET/plugins/METTesterPostProcessorHarvesting.h"

METTesterPostProcessorHarvesting::METTesterPostProcessorHarvesting(const edm::ParameterSet &iConfig) {
  runDir = iConfig.getUntrackedParameter<std::string>("runDir");
  inputMETLabelRECO_ = iConfig.getParameter<edm::InputTag>("METTypeRECO");
  inputMETLabelMiniAOD_ = iConfig.getParameter<edm::InputTag>("METTypeMiniAOD");
}

METTesterPostProcessorHarvesting::~METTesterPostProcessorHarvesting() {}

// ------------ method called right after a run ends ------------
void METTesterPostProcessorHarvesting::dqmEndJob(DQMStore::IBooker &ibook_, DQMStore::IGetter &iget_) {
  std::vector<std::string> subDirVec;
  iget_.setCurrentFolder(runDir);
  met_dirs = iget_.getSubdirs();
  bool found_reco_dir = false;
  bool found_miniaod_dir = false;

  // loop over met subdirectories
  for (int i = 0; i < int(met_dirs.size()); i++) {
    if (met_dirs[i] == (runDir + inputMETLabelRECO_.label())) {
      found_reco_dir = true;
    }
    if (met_dirs[i] == (runDir + inputMETLabelMiniAOD_.label())) {
      found_miniaod_dir = true;
    }
  }
  if (found_miniaod_dir && found_reco_dir) {
    std::string rundir_reco = runDir + inputMETLabelRECO_.label();
    std::string rundir_miniaod = runDir + inputMETLabelMiniAOD_.label();
    MonitorElement *mMET_Reco = iget_.get(rundir_reco + "/" + "MET");
    MonitorElement *mMETPhi_Reco = iget_.get(rundir_reco + "/" + "METPhi");
    MonitorElement *mSumET_Reco = iget_.get(rundir_reco + "/" + "SumET");
    MonitorElement *mMETDiff_GenMETTrue_Reco = iget_.get(rundir_reco + "/" + "METDiff_GenMETTrue");
    MonitorElement *mMETDeltaPhi_GenMETTrue_Reco = iget_.get(rundir_reco + "/" + "METDeltaPhi_GenMETTrue");
    MonitorElement *mPFPhotonEtFraction_Reco = iget_.get(rundir_reco + "/" + "photonEtFraction");
    MonitorElement *mPFNeutralHadronEtFraction_Reco = iget_.get(rundir_reco + "/" + "neutralHadronEtFraction");
    MonitorElement *mPFChargedHadronEtFraction_Reco = iget_.get(rundir_reco + "/" + "chargedHadronEtFraction");
    MonitorElement *mPFHFHadronEtFraction_Reco = iget_.get(rundir_reco + "/" + "HFHadronEtFraction");
    MonitorElement *mPFHFEMEtFraction_Reco = iget_.get(rundir_reco + "/" + "HFEMEtFraction");
    MonitorElement *mMETDiff_GenMETTrue_MET20to40_Reco = iget_.get(rundir_reco + "/" + "METDiff_GenMETTrue_MET20to40");
    MonitorElement *mMETDiff_GenMETTrue_MET100to150_Reco =
        iget_.get(rundir_reco + "/" + "METDiff_GenMETTrue_MET100to150");
    MonitorElement *mMETDiff_GenMETTrue_MET300to400_Reco =
        iget_.get(rundir_reco + "/" + "METDiff_GenMETTrue_MET300to400");

    std::vector<MonitorElement *> ME_Reco;
    ME_Reco.push_back(mMET_Reco);
    ME_Reco.push_back(mMETPhi_Reco);
    ME_Reco.push_back(mSumET_Reco);
    ME_Reco.push_back(mMETDiff_GenMETTrue_Reco);
    ME_Reco.push_back(mMETDeltaPhi_GenMETTrue_Reco);
    ME_Reco.push_back(mPFPhotonEtFraction_Reco);
    ME_Reco.push_back(mPFNeutralHadronEtFraction_Reco);
    ME_Reco.push_back(mPFChargedHadronEtFraction_Reco);
    ME_Reco.push_back(mPFHFHadronEtFraction_Reco);
    ME_Reco.push_back(mPFHFEMEtFraction_Reco);
    ME_Reco.push_back(mMETDiff_GenMETTrue_MET20to40_Reco);
    ME_Reco.push_back(mMETDiff_GenMETTrue_MET100to150_Reco);
    ME_Reco.push_back(mMETDiff_GenMETTrue_MET300to400_Reco);

    MonitorElement *mMET_MiniAOD = iget_.get(rundir_miniaod + "/" + "MET1");
    MonitorElement *mMETPhi_MiniAOD = iget_.get(rundir_miniaod + "/" + "METPhi");
    MonitorElement *mSumET_MiniAOD = iget_.get(rundir_miniaod + "/" + "SumET");
    MonitorElement *mMETDiff_GenMETTrue_MiniAOD = iget_.get(rundir_miniaod + "/" + "METDiff_GenMETTrue");
    MonitorElement *mMETDeltaPhi_GenMETTrue_MiniAOD = iget_.get(rundir_miniaod + "/" + "METDeltaPhi_GenMETTrue");
    MonitorElement *mPFPhotonEtFraction_MiniAOD = iget_.get(rundir_miniaod + "/" + "photonEtFraction");
    MonitorElement *mPFNeutralHadronEtFraction_MiniAOD = iget_.get(rundir_miniaod + "/" + "neutralHadronEtFraction");
    MonitorElement *mPFChargedHadronEtFraction_MiniAOD = iget_.get(rundir_miniaod + "/" + "chargedHadronEtFraction");
    MonitorElement *mPFHFHadronEtFraction_MiniAOD = iget_.get(rundir_miniaod + "/" + "HFHadronEtFraction");
    MonitorElement *mPFHFEMEtFraction_MiniAOD = iget_.get(rundir_miniaod + "/" + "HFEMEtFraction");
    MonitorElement *mMETDiff_GenMETTrue_MET20to40_MiniAOD =
        iget_.get(rundir_miniaod + "/" + "METResolution_GenMETTrue_MET20to40");
    MonitorElement *mMETDiff_GenMETTrue_MET100to150_MiniAOD =
        iget_.get(rundir_miniaod + "/" + "METResolution_GenMETTrue_MET100to150");
    MonitorElement *mMETDiff_GenMETTrue_MET300to400_MiniAOD =
        iget_.get(rundir_miniaod + "/" + "METResolution_GenMETTrue_MET300to400");

    std::vector<MonitorElement *> ME_MiniAOD;
    ME_MiniAOD.push_back(mMET_MiniAOD);
    ME_MiniAOD.push_back(mMETPhi_MiniAOD);
    ME_MiniAOD.push_back(mSumET_MiniAOD);
    ME_MiniAOD.push_back(mMETDiff_GenMETTrue_MiniAOD);
    ME_MiniAOD.push_back(mMETDeltaPhi_GenMETTrue_MiniAOD);
    ME_MiniAOD.push_back(mPFPhotonEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mPFNeutralHadronEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mPFChargedHadronEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mPFHFHadronEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mPFHFEMEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mMETDiff_GenMETTrue_MET20to40_MiniAOD);
    ME_MiniAOD.push_back(mMETDiff_GenMETTrue_MET100to150_MiniAOD);
    ME_MiniAOD.push_back(mMETDiff_GenMETTrue_MET300to400_MiniAOD);

    ibook_.setCurrentFolder(runDir + "MiniAOD_over_RECO");
    mMET_MiniAOD_over_Reco = ibook_.book1D("MET_MiniAOD_over_RECO", (TH1F *)mMET_Reco->getRootObject());
    mMETPhi_MiniAOD_over_Reco = ibook_.book1D("METPhi_MiniAOD_over_RECO", (TH1F *)mMETPhi_Reco->getRootObject());
    mSumET_MiniAOD_over_Reco = ibook_.book1D("SumET_MiniAOD_over_RECO", (TH1F *)mSumET_Reco->getRootObject());
    mMETDiff_GenMETTrue_MiniAOD_over_Reco =
        ibook_.book1D("METDiff_GenMETTrue_MiniAOD_over_RECO", (TH1F *)mMETDiff_GenMETTrue_Reco->getRootObject());
    mMETDeltaPhi_GenMETTrue_MiniAOD_over_Reco = ibook_.book1D("METDeltaPhi_GenMETTrue_MiniAOD_over_RECO",
                                                              (TH1F *)mMETDeltaPhi_GenMETTrue_Reco->getRootObject());
    mPFPhotonEtFraction_MiniAOD_over_Reco =
        ibook_.book1D("photonEtFraction_MiniAOD_over_RECO", (TH1F *)mPFPhotonEtFraction_Reco->getRootObject());
    mPFNeutralHadronEtFraction_MiniAOD_over_Reco = ibook_.book1D(
        "neutralHadronEtFraction_MiniAOD_over_RECO", (TH1F *)mPFNeutralHadronEtFraction_Reco->getRootObject());
    mPFChargedHadronEtFraction_MiniAOD_over_Reco = ibook_.book1D(
        "chargedHadronEtFraction_MiniAOD_over_RECO", (TH1F *)mPFChargedHadronEtFraction_Reco->getRootObject());
    mPFHFHadronEtFraction_MiniAOD_over_Reco =
        ibook_.book1D("HFHadronEtFraction_MiniAOD_over_RECO", (TH1F *)mPFHFHadronEtFraction_Reco->getRootObject());
    mPFHFEMEtFraction_MiniAOD_over_Reco =
        ibook_.book1D("HFEMEtEtFraction_MiniAOD_over_RECO", (TH1F *)mPFHFEMEtFraction_Reco->getRootObject());
    mMETDiff_GenMETTrue_MET20to40_MiniAOD_over_Reco =
        ibook_.book1D("METResolution_GenMETTrue_MET20to40_MiniAOD_over_RECO",
                      (TH1F *)mMETDiff_GenMETTrue_MET20to40_Reco->getRootObject());
    mMETDiff_GenMETTrue_MET100to150_MiniAOD_over_Reco =
        ibook_.book1D("METResolution_GenMETTrue_MET100to150_MiniAOD_over_RECO",
                      (TH1F *)mMETDiff_GenMETTrue_MET100to150_Reco->getRootObject());
    mMETDiff_GenMETTrue_MET300to400_MiniAOD_over_Reco =
        ibook_.book1D("METResolution_GenMETTrue_MET300to400_MiniAOD_over_RECO",
                      (TH1F *)mMETDiff_GenMETTrue_MET300to400_Reco->getRootObject());

    std::vector<MonitorElement *> ME_MiniAOD_over_Reco;
    ME_MiniAOD_over_Reco.push_back(mMET_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETPhi_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mSumET_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDiff_GenMETTrue_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDeltaPhi_GenMETTrue_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFPhotonEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFNeutralHadronEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFChargedHadronEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFHFHadronEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFHFEMEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDiff_GenMETTrue_MET20to40_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDiff_GenMETTrue_MET100to150_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDiff_GenMETTrue_MET300to400_MiniAOD_over_Reco);

    for (unsigned int j = 0; j < ME_MiniAOD_over_Reco.size(); j++) {
      MonitorElement *monReco = ME_Reco[j];
      if (monReco && monReco->getRootObject()) {
        MonitorElement *monMiniAOD = ME_MiniAOD[j];
        if (monMiniAOD && monMiniAOD->getRootObject()) {
          MonitorElement *monMiniAOD_over_RECO = ME_MiniAOD_over_Reco[j];
          if (monMiniAOD_over_RECO && monMiniAOD_over_RECO->getRootObject()) {
            for (int i = 0; i <= (monMiniAOD_over_RECO->getNbinsX() + 1); i++) {
              if (monReco->getBinContent(i) != 0) {
                monMiniAOD_over_RECO->setBinContent(i, monMiniAOD->getBinContent(i) / monReco->getBinContent(i));
              } else if (monMiniAOD->getBinContent(i) != 0) {
                monMiniAOD_over_RECO->setBinContent(i, -0.5);
              }
            }
          }
        }
      }
    }
  }
}

void METTesterPostProcessorHarvesting::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("runDir", "JetMET/METValidation");
  desc.add<edm::InputTag>("METTypeRECO", edm::InputTag("PfMetT1"));
  desc.add<edm::InputTag>("METTypeMiniAOD", edm::InputTag("slimmedMETs"));
  descriptions.addWithDefaultLabel(desc);
}

