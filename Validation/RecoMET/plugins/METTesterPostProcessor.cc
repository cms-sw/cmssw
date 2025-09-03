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
	for (std::string bt: {"MET", "Phi"}) { // loop over bin types
	  mMETDiffAggr[bt]  = ibook_.book1D("mMETDiffAggr_" + bt,  "mMETDiffAggr_" + bt,  mNBins[bt],
										std::visit([](const auto& arr) {return arr.data();}, mEdges[bt]));
	  mMETRespAggr[bt]  = ibook_.book1D("mMETRespAggr_" + bt,  "mMETRespAggr_" + bt,  mNBins[bt],
										std::visit([](const auto& arr) {return arr.data();}, mEdges[bt]));
	  mMETResolAggr[bt] = ibook_.book1D("mMETResolAggr_" + bt, "mMETResolAggr_" + bt, mNBins[bt],
										std::visit([](const auto& arr) {return arr.data();}, mEdges[bt]));
	  mMETSignAggr[bt]  = ibook_.book1D("mMETSignAggr_" + bt,  "mMETSignAggr_" + bt,  mNBins[bt],
										std::visit([](const auto& arr) {return arr.data();}, mEdges[bt]));
	}
    mFillAggrHistograms(met_dirs[i], iget_);
  }
}

void METTesterPostProcessor::mFillAggrHistograms(std::string metdir, DQMStore::IGetter &iget) {
  for (std::string bt: {"MET", "Phi"}) { // loop over bin types
	for (unsigned idx = 0; idx < mNBins[bt]-1; ++idx) {
	  std::string edges = METTester::binStr(mArrayIdx<float>(mEdges[bt], idx), mArrayIdx<float>(mEdges[bt], idx+1), true);
	  mArrayIdx<MElem*>(mMET[bt], idx)						= iget.get(metdir + "/MET_" + bt.c_str() + edges);
	  mArrayIdx<MElem*>(mMETDiff_GenMETTrue[bt], idx)		= iget.get(metdir + "/METDiff_GenMETTrue_" + bt.c_str() + edges);
	  mArrayIdx<MElem*>(mMETRatio_GenMETTrue[bt], idx)		= iget.get(metdir + "/METRatio_GenMETTrue_" + bt.c_str() + edges);
	  mArrayIdx<MElem*>(mMETDeltaPhi_GenMETTrue[bt], idx)	= iget.get(metdir + "/METDeltaPhi_GenMETTrue_" + bt.c_str() + edges);
	}

	// check one object, if it exists, then the remaining ME's exists too
	// for genmet none of these ME's are filled
	if (mArrayIdx<MElem*>(mMETDiff_GenMETTrue[bt], 0) && mArrayIdx<MElem*>(mMETDiff_GenMETTrue[bt], 0)->getRootObject()) {
	  for (unsigned idx = 0; idx < mNBins[bt]-1; ++idx) {
		mMETDiffAggr[bt]->setBinContent(idx+1, mArrayIdx<MElem*>(mMETDiff_GenMETTrue[bt], idx)->getMean());
		mMETDiffAggr[bt]->setBinError(idx + 1, mArrayIdx<MElem*>(mMETDiff_GenMETTrue[bt], idx)->getRMS());

		float ratioMean = mArrayIdx<MElem*>(mMETRatio_GenMETTrue[bt], idx)->getMean();
		float ratioRMS = mArrayIdx<MElem*>(mMETRatio_GenMETTrue[bt], idx)->getRMS();
		mMETRespAggr[bt]->setBinContent(idx+1, ratioMean);
		mMETRespAggr[bt]->setBinError(idx+1, ratioRMS);

		float metMean = mArrayIdx<MElem*>(mMET[bt], idx)->getMean();
		float metRMS = mArrayIdx<MElem*>(mMET[bt], idx)->getRMS();
		float resolError = mArrayIdx<MElem*>(mMET[bt], idx)->getRMSError();
		mMETResolAggr[bt]->setBinContent(idx + 1, metRMS);
		mMETResolAggr[bt]->setBinError(idx + 1, resolError);

		float significance = metMean / metRMS;
		mMETSignAggr[bt]->setBinContent(idx+1, significance);
		mMETSignAggr[bt]->setBinError(idx + 1,
									  significance * std::sqrt((metRMS * metRMS / (metMean * metMean)) +
															   (resolError * resolError / (metRMS * metRMS))));
	  }
	}
  }
}

void METTesterPostProcessor::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("isHLT", false);
  descriptions.addWithDefaultLabel(desc);
}
