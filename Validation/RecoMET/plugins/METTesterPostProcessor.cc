// -*- C++ -*-
//
// Package:    Validation/RecoMET
// Class:      METTesterPostProcessor
//
// Original Author:  "Matthias Weber"
//         Created:  Sun Feb 22 14:35:25 CET 2015
//
#include "Validation/RecoMET/plugins/METTesterPostProcessor.h"

METTesterPostProcessor::METTesterPostProcessor(const edm::ParameterSet &iConfig) {
  isHLT = iConfig.getUntrackedParameter<bool>("isHLT", false);
}
METTesterPostProcessor::~METTesterPostProcessor() {}

// ------------ method called right after a run ends ------------
void METTesterPostProcessor::dqmEndJob(DQMStore::IBooker &ibook_, DQMStore::IGetter &iget_) {
  std::vector<std::string> subDirVec;
  std::string RunDir;
  if (isHLT)
    RunDir = "HLT/JetMET/METValidation/";
  else
    RunDir = "JetMET/METValidation/";

  iget_.setCurrentFolder(RunDir);
  met_dirs = iget_.getSubdirs();

  // loop over met subdirectories
  for (size_t i = 0; i < met_dirs.size(); i++) {
    ibook_.setCurrentFolder(met_dirs[i]);
    mMETDiffAggr_METBins = ibook_.book1D("mMETDiffAggr_METBins", "mMETDiffAggr_METBins", mNMETBins, mMETBins.data());
    mMETDiffAggr_EtaBins = ibook_.book1D("mMETDiffAggr_EtaBins", "mMETDiffAggr_EtaBins", mNEtaBins, mEtaBins.data());
    mMETDiffAggr_PhiBins = ibook_.book1D("mMETDiffAggr_PhiBins", "mMETDiffAggr_PhiBins", mNPhiBins, mPhiBins.data());
    mMETRespAggr_METBins = ibook_.book1D("mMETRespAggr_METBins", "mMETRespAggr_METBins", mNMETBins, mMETBins.data());
    mMETRespAggr_EtaBins = ibook_.book1D("mMETRespAggr_EtaBins", "mMETRespAggr_EtaBins", mNEtaBins, mEtaBins.data());
    mMETRespAggr_PhiBins = ibook_.book1D("mMETRespAggr_PhiBins", "mMETRespAggr_PhiBins", mNPhiBins, mPhiBins.data());
    mMETResolAggr_METBins = ibook_.book1D("mMETResolAggr_METBins", "mMETResolAggr_METBins", mNMETBins, mMETBins.data());
    mMETResolAggr_EtaBins = ibook_.book1D("mMETResolAggr_EtaBins", "mMETResolAggr_EtaBins", mNEtaBins, mEtaBins.data());
    mMETResolAggr_PhiBins = ibook_.book1D("mMETResolAggr_PhiBins", "mMETResolAggr_PhiBins", mNPhiBins, mPhiBins.data());
    mMETSignAggr_METBins = ibook_.book1D("mMETSignAggr_METBins", "mMETSignAggr_METBins", mNMETBins, mMETBins.data());
    mMETSignAggr_EtaBins = ibook_.book1D("mMETSignAggr_EtaBins", "mMETSignAggr_EtaBins", mNEtaBins, mEtaBins.data());
    mMETSignAggr_PhiBins = ibook_.book1D("mMETSignAggr_PhiBins", "mMETSignAggr_PhiBins", mNPhiBins, mPhiBins.data());
    mFillAggrHistograms(met_dirs[i], iget_);
  }
}

void METTesterPostProcessor::mFillAggrHistograms(std::string metdir, DQMStore::IGetter &iget) {
  for (unsigned metIdx = 0; metIdx < mNMETBins - 1; ++metIdx) {
    std::string edges = METTester::binStr(mMETBins[metIdx], mMETBins[metIdx + 1], true);
    mMET_METBins[metIdx] = iget.get(metdir + "/MET_MET" + edges);
    mMETDiff_GenMETTrue_METBins[metIdx] = iget.get(metdir + "/METDiff_GenMETTrue_MET" + edges);
    mMETRatio_GenMETTrue_METBins[metIdx] = iget.get(metdir + "/METRatio_GenMETTrue_MET" + edges);
    mMETDeltaPhi_GenMETTrue_METBins[metIdx] = iget.get(metdir + "/METDeltaPhi_GenMETTrue_MET" + edges);
  }
  for (unsigned metIdx = 0; metIdx < mNEtaBins - 1; ++metIdx) {
    std::string edges = METTester::binStr(mEtaBins[metIdx], mEtaBins[metIdx + 1]);
    mMET_EtaBins[metIdx] = iget.get(metdir + "/MET_Eta" + edges);
    mMETDiff_GenMETTrue_EtaBins[metIdx] = iget.get(metdir + "/METDiff_GenMETTrue_Eta" + edges);
    mMETRatio_GenMETTrue_EtaBins[metIdx] = iget.get(metdir + "/METRatio_GenMETTrue_Eta" + edges);
    mMETDeltaPhi_GenMETTrue_EtaBins[metIdx] = iget.get(metdir + "/METDeltaPhi_GenMETTrue_Eta" + edges);
  }
  for (unsigned metIdx = 0; metIdx < mNPhiBins - 1; ++metIdx) {
    std::string edges = METTester::binStr(mPhiBins[metIdx], mPhiBins[metIdx + 1]);
    mMET_PhiBins[metIdx] = iget.get(metdir + "/MET_Phi" + edges);
    mMETDiff_GenMETTrue_PhiBins[metIdx] = iget.get(metdir + "/METDiff_GenMETTrue_Phi" + edges);
    mMETRatio_GenMETTrue_PhiBins[metIdx] = iget.get(metdir + "/METRatio_GenMETTrue_Phi" + edges);
    mMETDeltaPhi_GenMETTrue_PhiBins[metIdx] = iget.get(metdir + "/METDeltaPhi_GenMETTrue_Phi" + edges);
  }

  // check one object, if existing, then the remaining ME's exist too
  // for genmet none of these ME's are filled
  if (mMETDiff_GenMETTrue_METBins[0] && mMETDiff_GenMETTrue_METBins[0]->getRootObject()) {
    for (unsigned metIdx = 0; metIdx < mNMETBins - 1; ++metIdx) {
      mMETDiffAggr_METBins->setBinContent(metIdx + 1, mMETDiff_GenMETTrue_METBins[metIdx]->getMean());
      mMETDiffAggr_METBins->setBinError(metIdx + 1, mMETDiff_GenMETTrue_METBins[metIdx]->getRMS());

      float ratioMean = mMETRatio_GenMETTrue_METBins[metIdx]->getMean();
      float ratioRMS = mMETRatio_GenMETTrue_METBins[metIdx]->getRMS();
      mMETRespAggr_METBins->setBinContent(metIdx + 1, ratioMean);
      mMETRespAggr_METBins->setBinError(metIdx + 1, ratioRMS);

      float metMean = mMET_METBins[metIdx]->getMean();
      float metRMS = mMET_METBins[metIdx]->getRMS();
      float resol = mMET_METBins[metIdx]->getRMS() / ratioMean;
      float resolError = metRMS * ratioRMS * ratioRMS / (ratioMean * ratioMean);
      mMETResolAggr_METBins->setBinContent(metIdx + 1, resol);
      mMETResolAggr_METBins->setBinError(metIdx + 1, resolError);

      float significance = mMET_METBins[metIdx]->getMean() / resol;
      mMETSignAggr_METBins->setBinContent(metIdx + 1, significance);
      mMETSignAggr_METBins->setBinError(metIdx + 1,
                                        significance * std::sqrt((metRMS * metRMS / (metMean * metMean)) +
                                                                 (resolError * resolError / (resol * resol))));
    }
  }

  if (mMETDiff_GenMETTrue_EtaBins[0] && mMETDiff_GenMETTrue_EtaBins[0]->getRootObject()) {
    for (unsigned metIdx = 0; metIdx < mNEtaBins - 1; ++metIdx) {
      mMETDiffAggr_EtaBins->setBinContent(metIdx + 1, mMETDiff_GenMETTrue_EtaBins[metIdx]->getMean());
      mMETDiffAggr_EtaBins->setBinError(metIdx + 1, mMETDiff_GenMETTrue_EtaBins[metIdx]->getRMS());

      float ratioMean = mMETRatio_GenMETTrue_EtaBins[metIdx]->getMean();
      float ratioRMS = mMETRatio_GenMETTrue_EtaBins[metIdx]->getRMS();
      mMETRespAggr_EtaBins->setBinContent(metIdx + 1, ratioMean);
      mMETRespAggr_EtaBins->setBinError(metIdx + 1, ratioRMS);

      float metMean = mMET_EtaBins[metIdx]->getMean();
      float metRMS = mMET_EtaBins[metIdx]->getRMS();
      float resol = mMET_EtaBins[metIdx]->getRMS() / ratioMean;
      float resolError = metRMS * ratioRMS * ratioRMS / (ratioMean * ratioMean);
      mMETResolAggr_EtaBins->setBinContent(metIdx + 1, resol);
      mMETResolAggr_EtaBins->setBinError(metIdx + 1, resolError);

      float significance = mMET_EtaBins[metIdx]->getMean() / resol;
      mMETSignAggr_EtaBins->setBinContent(metIdx + 1, significance);
      mMETSignAggr_EtaBins->setBinError(metIdx + 1,
                                        significance * std::sqrt((metRMS * metRMS / (metMean * metMean)) +
                                                                 (resolError * resolError / (resol * resol))));
    }
  }

  if (mMETDiff_GenMETTrue_PhiBins[0] && mMETDiff_GenMETTrue_PhiBins[0]->getRootObject()) {
    for (unsigned metIdx = 0; metIdx < mNPhiBins - 1; ++metIdx) {
      mMETDiffAggr_PhiBins->setBinContent(metIdx + 1, mMETDiff_GenMETTrue_PhiBins[metIdx]->getMean());
      mMETDiffAggr_PhiBins->setBinError(metIdx + 1, mMETDiff_GenMETTrue_PhiBins[metIdx]->getRMS());

      float ratioMean = mMETRatio_GenMETTrue_PhiBins[metIdx]->getMean();
      float ratioRMS = mMETRatio_GenMETTrue_PhiBins[metIdx]->getRMS();
      mMETRespAggr_PhiBins->setBinContent(metIdx + 1, ratioMean);
      mMETRespAggr_PhiBins->setBinError(metIdx + 1, ratioRMS);

      float metMean = mMET_PhiBins[metIdx]->getMean();
      float metRMS = mMET_PhiBins[metIdx]->getRMS();
      float resol = mMET_PhiBins[metIdx]->getRMS() / ratioMean;
      float resolError = metRMS * ratioRMS * ratioRMS / (ratioMean * ratioMean);
      mMETResolAggr_PhiBins->setBinContent(metIdx + 1, resol);
      mMETResolAggr_PhiBins->setBinError(metIdx + 1, resolError);

      float significance = mMET_PhiBins[metIdx]->getMean() / resol;
      mMETSignAggr_PhiBins->setBinContent(metIdx + 1, significance);
      mMETSignAggr_PhiBins->setBinError(metIdx + 1,
                                        significance * std::sqrt((metRMS * metRMS / (metMean * metMean)) +
                                                                 (resolError * resolError / (resol * resol))));
    }
  }
}

void METTesterPostProcessor::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("isHLT", false);
  descriptions.addWithDefaultLabel(desc);
}
