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

	// differences
	mMETDiffAggr["MET"] = ibook_.book1D("METDiffAggr_MET", "METDiffAggr_MET", mNMETBins, mMETBins.data());
	mMETDiffAggr["Phi"] = ibook_.book1D("METDiffAggr_Phi", "METDiffAggr_Phi", mNPhiBins, mPhiBins.data());
	mMETDeltaPhiAggr["MET"] = ibook_.book1D("METDeltaPhiAggr_MET", "METDeltaPhiAggr_MET", mNMETBins, mMETBins.data());
	mMETDeltaPhiAggr["Phi"] = ibook_.book1D("METDeltaPhiAggr_Phi", "METDeltaPhiAggr_Phi", mNPhiBins, mPhiBins.data());

	// response
	mMETRespAggr["MET"] = ibook_.book1D("METRespAggr_MET", "METRespAggr_MET", mNMETBins, mMETBins.data());
	mMETRespAggr["Phi"] = ibook_.book1D("METRespAggr_Phi", "METRespAggr_Phi", mNPhiBins, mPhiBins.data());
	
	// resolution
	mMETResolAggr["Phi"] = ibook_.book1D("METResolAggr_Phi", "METResolAggr_Phi", mNPhiBins, mPhiBins.data());
	mMETGenResolAggr["Phi"] = ibook_.book1D("METGenResolAggr_Phi", "METGenResolAggr_Phi", mNPhiBins, mPhiBins.data());
	mMETResolDiffAggr["Phi"] = ibook_.book1D("METResolDiffAggr_Phi", "METResolDiffAggr_Phi", mNPhiBins, mPhiBins.data());
	
	// significance
	mMETSignAggr["Phi"] = ibook_.book1D("METSignAggr_Phi", "METSignAggr_Phi", mNPhiBins, mPhiBins.data());
	mMETGenSignAggr["Phi"] = ibook_.book1D("METGenSignAggr_Phi", "METGenSignAggr_Phi", mNPhiBins, mPhiBins.data());
	mMETSignDiffAggr["Phi"] = ibook_.book1D("METSignDiffAggr_Phi", "METSignDiffAggr_Phi", mNPhiBins, mPhiBins.data());
	
	mFillAggrHistograms(met_dirs[i], iget_);
  }
}

bool METTesterPostProcessor::mCheckHisto(MElem *h) { return h && h->getRootObject(); }

std::vector<std::unordered_map<std::string,float>> METTesterPostProcessor::projectionMeanAndRMS(MElem* src,
																								const std::vector<float>& bins, std::string axis) {
  std::vector<std::unordered_map<std::string,float>> ret;
  TH2F* h2 = src->getTH2F();

  if (axis == "X") {
	assert(h2->GetXaxis()->GetXmin() <= bins[0] and h2->GetXaxis()->GetXmax() >= bins[bins.size()-1]);
  }
  else if (axis == "Y") {
	assert(h2->GetYaxis()->GetXmin() <= bins[0] and h2->GetYaxis()->GetXmax() >= bins[bins.size()-1]);
  }
  else {
	edm::LogWarning("METTesterPostProcessor") << "Unsupported option axis=" << axis << ".";
  }
  
  for (unsigned binId=0; binId<bins.size()-1; binId++) {
	TH1D* proj = nullptr;
	if (axis == "X") {
	  int binIdLeft = h2->GetXaxis()->FindBin(bins[binId]);
	  int binIdRight = h2->GetXaxis()->FindBin(bins[binId+1]);
	  // std::cout << bins[binId] << ", " << bins[binId+1] << std::endl;
	  // std::cout << binIdLeft << ", " << binIdRight << std::endl;
	  proj = h2->ProjectionY(("MeanAndRMS" + std::to_string(binId)).c_str(), binIdLeft, binIdRight);
	}
	else if (axis == "Y") {
	  int binIdBottom = h2->GetYaxis()->FindBin(bins[binId]);
	  int binIdUp = h2->GetYaxis()->FindBin(bins[binId+1]);
	  // std::cout << bins[binId] << ", " << bins[binId+1] << std::endl;
	  // std::cout << binIdBottom << ", " << binIdUp << std::endl;
	  proj = h2->ProjectionX(("MeanAndRMS" + std::to_string(binId)).c_str(), binIdBottom, binIdUp);
	}
	// std::cout << proj->GetMean() << ", " << proj->GetRMS() << ", " << proj->GetMeanError() << ", " << proj->GetRMSError() << std::endl;
	// std::cout << "=========" << std::endl;
	std::unordered_map<std::string,float> tmp = {
	  {"mean", proj->GetMean()},
	  {"meanerr", proj->GetMeanError()},
	  {"rms", proj->GetRMS()},
	  {"rmserr", proj->GetRMSError()}
	};
	ret.push_back(tmp);
  }
  return ret;
}

void METTesterPostProcessor::fillProjectionHisto(MElem* src, MElem* dest, const std::vector<float>& bins) {
  TH2F* h2 = src->getTH2F();
  for (unsigned binId=0; binId<bins.size()-1; binId++) {
	int binIdLeft = h2->GetXaxis()->FindBin(bins[binId]);
	int binIdRight = h2->GetXaxis()->FindBin(bins[binId+1]);
	TH1D* projY = h2->ProjectionY((std::to_string(binId)).c_str(), binIdLeft, binIdRight);
	dest->setBinContent(binId, projY->GetMean());
	dest->setBinError(binId, projY->GetRMS());
	delete projY;
  }
}

void METTesterPostProcessor::mFillAggrHistograms(std::string metdir, DQMStore::IGetter &iget) {
  mGenMETTrue_vs_MET		 = iget.get(metdir + "/GenMETTruevsMET");
  mGenMETPhi_vs_MET			 = iget.get(metdir + "/GenMETPhivsMET");
  mGenMETTrue_vs_mGenMETPhi	 = iget.get(metdir + "/GenMETTruevsGenMETPhi");
  mMETDiff_vs_GenMETTrue	 = iget.get(metdir + "/METDiffvsGenMETTrue");
  mMETDiff_vs_GenMETPhi  	 = iget.get(metdir + "/METDiffvsGenMETPhi");
  mMETRatio_vs_GenMETTrue	 = iget.get(metdir + "/METRatiovsGenMETTrue");
  mMETRatio_vs_GenMETPhi	 = iget.get(metdir + "/METRatiovsGenMETPhi");
  mMETDeltaPhi_vs_GenMETTrue = iget.get(metdir + "/METDeltaPhivsGenMETTrue");
  mMETDeltaPhi_vs_GenMETPhi  = iget.get(metdir + "/METDeltaPhivsGenMETPhi");

  // check if object exists
  // for genmet none of these ME's are filled
  if (!mCheckHisto(mGenMETTrue_vs_MET))				 logDebugEmptyHisto("mGenMETTrue_vs_MET");
  else if (!mCheckHisto(mGenMETPhi_vs_MET))			 logDebugEmptyHisto("mGenMETPhi_vs_MET");
  else if (!mCheckHisto(mGenMETTrue_vs_mGenMETPhi))	 logDebugEmptyHisto("mGenMETTrue_vs_mGenMETPhi");
  else if (!mCheckHisto(mMETDiff_vs_GenMETTrue))	 logDebugEmptyHisto("mMETDiff_vs_GenMETTrue");
  else if (!mCheckHisto(mMETDiff_vs_GenMETPhi))		 logDebugEmptyHisto("mMETDiff_vs_GenMETPhi");
  else if (!mCheckHisto(mMETRatio_vs_GenMETTrue))	 logDebugEmptyHisto("mMETRatio_vs_GenMETTrue");
  else if (!mCheckHisto(mMETRatio_vs_GenMETPhi))	 logDebugEmptyHisto("mMETRatio_vs_GenMETPhi");
  else if (!mCheckHisto(mMETDeltaPhi_vs_GenMETTrue)) logDebugEmptyHisto("mMETDeltaPhi_vs_GenMETTrue");
  else if (!mCheckHisto(mMETDeltaPhi_vs_GenMETPhi))  logDebugEmptyHisto("mMETDeltaPhi_vs_GenMETPhi");

  // log histograms with zero entries
  if (mGenMETTrue_vs_MET->getEntries() < mEpsilonDouble)
	LogDebug("METTesterPostProcessor") << "Gen MET vs MET entries: " << mGenMETTrue_vs_MET->getEntries();
  else if (mGenMETPhi_vs_MET->getEntries() < mEpsilonDouble)
	LogDebug("METTesterPostProcessor") << "Gen MET Phi vs MET entries: " << mGenMETPhi_vs_MET->getEntries();
  else if (mGenMETTrue_vs_mGenMETPhi->getEntries() < mEpsilonDouble)
	LogDebug("METTesterPostProcessor") << "Gen MET vs Gen MET Phi entries: " << mGenMETTrue_vs_mGenMETPhi->getEntries();
  else if (mMETDiff_vs_GenMETTrue->getEntries() < mEpsilonDouble)
	LogDebug("METTesterPostProcessor") << "MET Diff vs Gen MET entries: " << mMETDiff_vs_GenMETTrue->getEntries();
  // else if (mMETDiff_vs_GenMETPhi->getEntries() < mEpsilonDouble)
  // 	LogDebug("METTesterPostProcessor") << "MET Diff vs Gen MET Phi entries: " << mMETDiff_vs_GenMETPhi->getEntries();
  // else if (mMETRatio_vs_GenMETTrue->getEntries() < mEpsilonDouble)
  // 	LogDebug("METTesterPostProcessor") << "MET Ratio vs Gen MET entries: " << mMETRatio_vs_GenMETTrue->getEntries();
  // else if (mMETRatio_vs_GenMETPhi->getEntries() < mEpsilonDouble)
  // 	LogDebug("METTesterPostProcessor") << "MET Ratio vs Gen MET Phi: " << mMETRatio_vs_GenMETPhi->getEntries();
  // else if (mMETDeltaPhi_vs_GenMETTrue->getEntries() < mEpsilonDouble)
  // 	LogDebug("METTesterPostProcessor") << "MET Delta Phi vs Gen MET entries: " << mMETDeltaPhi_vs_GenMETTrue->getEntries();
  // else if (mMETDeltaPhi_vs_GenMETPhi->getEntries() < mEpsilonDouble)
  // 	LogDebug("METTesterPostProcessor") << "MET Delta Phi vs Gen Phi: " << mMETDeltaPhi_vs_GenMETPhi->getEntries();

  // fillProjectionHisto(mMETDiff_vs_GenMETTrue, mMETDiffAggr["MET"], mMETBins);
  // fillProjectionHisto(mMETDeltaPhi_vs_GenMETTrue, mMETDeltaPhiAggr["MET"], mMETBins);
  // fillProjectionHisto(mMETDiff_vs_GenMETPhi, mMETDiffAggr["Phi"], mPhiBins);
  // fillProjectionHisto(mMETDeltaPhi_vs_GenMETPhi, mMETDeltaPhiAggr["Phi"], mPhiBins);

  // fillProjectionHisto(mMETRatio_vs_GenMETTrue, mMETRespAggr["MET"], mMETBins);
  // fillProjectionHisto(mMETRatio_vs_GenMETPhi, mMETRespAggr["Phi"], mPhiBins);

  // std::vector<std::unordered_map<std::string,float>> metRecoStats = projectionMeanAndRMS(mGenMETPhi_vs_MET, mMETBins, "Y");
  // std::vector<std::unordered_map<std::string,float>> metGenStats = projectionMeanAndRMS(mGenMETTrue_vs_mGenMETPhi, mMETBins);
  // for (unsigned binId=1; binId<=mNPhiBins; binId++) {
  // 	// reconstructed MET
  // 	const float& metMean   = metRecoStats[binId]["mean"];
  // 	const float& metRMS	   = metRecoStats[binId]["rms"];
  // 	const float& metRMSErr = metRecoStats[binId]["rmserror"];
  // 	mMETResolAggr["Phi"]->setBinContent(binId, metRMS);
  // 	mMETResolAggr["Phi"]->setBinError(binId, metRMSErr);

  // 	float significance = metRMS < mEpsilonFloat ? 0.f : metMean / metRMS;
  // 	float significanceErr = ( (metRMS < mEpsilonFloat or metMean < mEpsilonFloat)
  // 							  ? 0.f : mComputeSignErr(significance, metRMS, metMean, metRMSErr) );
  // 	mMETSignAggr["Phi"]->setBinContent(binId, significance);
  // 	mMETSignAggr["Phi"]->setBinError(binId, significanceErr);

  // 	// generated MET
  // 	const float& metGenMean		= metGenStats[binId]["mean"];
  // 	const float& metGenRMS		= metGenStats[binId]["rms"];
  // 	const float& metGenRMSErr	= metGenStats[binId]["rmserror"];
  // 	mMETGenResolAggr["Phi"]->setBinContent(binId, metGenRMS);
  // 	mMETGenResolAggr["Phi"]->setBinError(binId, metGenRMSErr);

  // 	float significanceGen = metGenRMS < mEpsilonFloat ? 0.f : metGenMean / metGenRMS;
  // 	float significanceGenErr = ( (metGenRMS < mEpsilonFloat or metGenMean < mEpsilonFloat)
  // 								 ? 0.f : mComputeSignErr(significanceGen, metGenRMS, metGenMean, metGenRMSErr) );
  // 	mMETGenSignAggr["Phi"]->setBinContent(binId, significanceGen);
  // 	mMETGenSignAggr["Phi"]->setBinError(binId, significanceGenErr);

  // 	// differences
  // 	mMETResolDiffAggr["Phi"]->setBinContent(binId, metRMS - metGenRMS);
  // 	mMETResolDiffAggr["Phi"]->setBinError(binId, std::sqrt(metGenRMSErr*metGenRMSErr + metRMSErr*metRMSErr));
  // 	mMETSignDiffAggr["Phi"]->setBinContent(binId, significance - significanceGen);
  // 	mMETSignDiffAggr["Phi"]->setBinError(binId, std::sqrt(significanceErr*significanceErr + significanceGenErr*significanceGenErr));
  // }
}

// Compute significance error
float METTesterPostProcessor::mComputeSignErr(float significance, float metRMS, float metMean, float metRMSError) {
  return significance * std::sqrt((metRMS * metRMS / (metMean * metMean)) +
								  (metRMSError * metRMSError / (metRMS * metRMS)));
}

void METTesterPostProcessor::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("runDir", "JetMET/METValidation");
  descriptions.addWithDefaultLabel(desc);
}

void METTesterPostProcessor::logDebugEmptyHisto(std::string hname) {
  LogDebug("METTesterPostProcessor") << "Histogram " << hname << " is empty.";
  return;
}
