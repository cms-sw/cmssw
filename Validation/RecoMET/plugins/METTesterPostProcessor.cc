#include "Validation/RecoMET/plugins/METTesterPostProcessor.h"

METTesterPostProcessor::METTesterPostProcessor(const edm::ParameterSet &iConfig) {
  runDir = iConfig.getUntrackedParameter<std::string>("runDir");
}
METTesterPostProcessor::~METTesterPostProcessor() {}

// ------------ method called right after a run ends ------------
void METTesterPostProcessor::dqmEndJob(DQMStore::IBooker &ibook_, DQMStore::IGetter &iget_) {
  std::vector<std::string> subDirVec;
  std::string RunDir = runDir;
  iget_.setCurrentFolder(RunDir);
  met_dirs = iget_.getSubdirs();

  // loop over met subdirectories
  for (size_t i = 0; i < met_dirs.size(); i++) {
    ibook_.setCurrentFolder(met_dirs[i]);
    for (std::string bt : {"MET", "Phi"}) {  // loop over bin types
      mMETDiffAggr[bt] = ibook_.book1D("METDiffAggr_" + bt,
                                       "METDiffAggr_" + bt,
                                       mNBins[bt],
                                       std::visit([](const auto &arr) { return arr.data(); }, mEdges[bt]));
      mMETRespAggr[bt] = ibook_.book1D("METRespAggr_" + bt,
                                       "METRespAggr_" + bt,
                                       mNBins[bt],
                                       std::visit([](const auto &arr) { return arr.data(); }, mEdges[bt]));
      mMETResolAggr[bt] = ibook_.book1D("METResolAggr_" + bt,
                                        "METResolAggr_" + bt,
                                        mNBins[bt],
                                        std::visit([](const auto &arr) { return arr.data(); }, mEdges[bt]));
      mMETSignAggr[bt] = ibook_.book1D("METSignAggr_" + bt,
                                       "METSignAggr_" + bt,
                                       mNBins[bt],
                                       std::visit([](const auto &arr) { return arr.data(); }, mEdges[bt]));
    }
    mFillAggrHistograms(met_dirs[i], iget_);
  }
}

bool METTesterPostProcessor::mCheckHisto(MElem *h) { return h && h->getRootObject(); }

void METTesterPostProcessor::mFillAggrHistograms(std::string metdir, DQMStore::IGetter &iget) {
  for (std::string bt : {"MET", "Phi"}) {  // loop over bin types
    for (unsigned idx = 0; idx < mNBins[bt]; ++idx) {
      std::string edges =
          METTester::binStr(mArrayIdx<float>(mEdges[bt], idx), mArrayIdx<float>(mEdges[bt], idx + 1), bt == "MET");
      mArrayIdx<MElem *>(mMET[bt], idx) = iget.get(metdir + "/MET_" + bt + edges);
      mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx) = iget.get(metdir + "/METDiff_GenMETTrue_" + bt + edges);
      mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx) = iget.get(metdir + "/METRatio_GenMETTrue_" + bt + edges);
      mArrayIdx<MElem *>(mMETDeltaPhi_GenMETTrue[bt], idx) = iget.get(metdir + "/METDeltaPhi_GenMETTrue_" + bt + edges);

      // check one object, if it exists, then the remaining ME's exists too
      // for genmet none of these ME's are filled
      if (mCheckHisto(mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], 0))) {
        // log histograms with zero entries
        if (mArrayIdx<MElem *>(mMET[bt], idx)->getEntries() < mEpsilonDouble ||
            mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx)->getEntries() < mEpsilonDouble ||
            mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx)->getEntries() < mEpsilonDouble ||
            mArrayIdx<MElem *>(mMETDeltaPhi_GenMETTrue[bt], idx)->getEntries() < mEpsilonDouble) {
          LogDebug("METTesterPostProcessor")
              << "At least one of the " << bt + edges << " histograms has zero entries:\n"
              << "  MET: " << mArrayIdx<MElem *>(mMET[bt], idx)->getEntries() << "\n"
              << "  METDiff: " << mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx)->getEntries() << "\n"
              << "  METRatio: " << mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx)->getEntries() << "\n"
              << "  METDeltaPhi: " << mArrayIdx<MElem *>(mMETDeltaPhi_GenMETTrue[bt], idx)->getEntries();
        }
      }
    }

    if (mCheckHisto(mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], 0))) {
      // compute and store MET quantities
      for (unsigned idx = 0; idx < mNBins[bt]; ++idx) {
        mMETDiffAggr[bt]->setBinContent(idx + 1, mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx)->getMean());
        mMETDiffAggr[bt]->setBinError(idx + 1, mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx)->getRMS());

        float ratioMean = mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx)->getMean();
        float ratioRMS = mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx)->getRMS();
        mMETRespAggr[bt]->setBinContent(idx + 1, ratioMean);
        mMETRespAggr[bt]->setBinError(idx + 1, ratioRMS);

        float metMean = mArrayIdx<MElem *>(mMET[bt], idx)->getMean();
        float metRMS = mArrayIdx<MElem *>(mMET[bt], idx)->getRMS();
        float resolError = mArrayIdx<MElem *>(mMET[bt], idx)->getRMSError();
        mMETResolAggr[bt]->setBinContent(idx + 1, metRMS);
        mMETResolAggr[bt]->setBinError(idx + 1, resolError);

        float significance = metRMS < mEpsilonFloat ? 0.f : metMean / metRMS;
        float significance_error = metRMS < mEpsilonFloat || metMean < mEpsilonFloat
                                       ? 0.f
                                       : significance * std::sqrt((metRMS * metRMS / (metMean * metMean)) +
                                                                  (resolError * resolError / (metRMS * metRMS)));
        mMETSignAggr[bt]->setBinContent(idx + 1, significance);
        mMETSignAggr[bt]->setBinError(idx + 1, significance_error);
      }
    }
  }
}

void METTesterPostProcessor::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("runDir", "JetMET/METValidation");
  descriptions.addWithDefaultLabel(desc);
}
