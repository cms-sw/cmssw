#include "Validation/RecoMET/plugins/METTesterPostProcessor.h"

#include <memory>

METTesterPostProcessor::METTesterPostProcessor(const edm::ParameterSet &iConfig) {
  runDir = iConfig.getUntrackedParameter<std::string>("runDir");
}
METTesterPostProcessor::~METTesterPostProcessor() {}

// ------------ method called right after a run ends ------------
void METTesterPostProcessor::dqmEndJob(DQMStore::IBooker &ibook_, DQMStore::IGetter &iget_) {
  iget_.setCurrentFolder(runDir);
  const std::vector<std::string> met_dirs = iget_.getSubdirs();

  // loop over met subdirectories; folders without the required inputs (gen MET, MET vs MHT, ...) are skipped
  for (const auto &metdir : met_dirs) {
    const auto inputs = mGetInputs(metdir, iget_);
    if (!inputs)
      continue;
    mFillAggrHistograms(metdir, *inputs, ibook_);
  }
}

std::optional<METTesterPostProcessor::Inputs> METTesterPostProcessor::mGetInputs(const std::string &metdir,
                                                                                 DQMStore::IGetter &iget) {
  std::vector<std::string> missing;
  auto get2D = [&](const std::string &name) -> MElem * {
    MElem *me = iget.get(metdir + "/" + name);
    if (!me or me->kind() != MonitorElement::Kind::TH2F) {
      missing.push_back(name);
      return nullptr;
    }
    if (me->getEntries() < 1)
      LogDebug("METTesterPostProcessor") << "Histogram " << metdir << "/" << name << " is empty.";
    return me;
  };

  Inputs in;
  in.genMETPhi_vs_MET = get2D("GenMETPhivsMET");
  in.genMETTrue_vs_GenMETPhi = get2D("GenMETTruevsGenMETPhi");
  in.metDiff_vs_GenMETTrue = get2D("METDiffvsGenMETTrue");
  in.metDiff_vs_GenMETPhi = get2D("METDiffvsGenMETPhi");
  in.metRatio_vs_GenMETTrue = get2D("METRatiovsGenMETTrue");
  in.metRatio_vs_GenMETPhi = get2D("METRatiovsGenMETPhi");
  in.metDeltaPhi_vs_GenMETTrue = get2D("METDeltaPhivsGenMETTrue");
  in.metDeltaPhi_vs_GenMETPhi = get2D("METDeltaPhivsGenMETPhi");

  if (!missing.empty()) {
    std::string names;
    for (const auto &m : missing)
      names += " " + m;
    edm::LogInfo("METTesterPostProcessor") << "Skipping folder " << metdir << ", missing histograms:" << names;
    return std::nullopt;
  }

  return in;
}

std::vector<METTesterPostProcessor::ProjStats> METTesterPostProcessor::projectionMeanAndRMS(
    MElem *src, const std::vector<float> &bins, const std::string &axis) {
  std::vector<ProjStats> ret;
  TH2F *h2 = src->getTH2F();

  if (axis != "X" and axis != "Y") {
    edm::LogWarning("METTesterPostProcessor") << "Unsupported option axis=" << axis << ".";
    return ret;
  }
  const TAxis *ax = axis == "X" ? h2->GetXaxis() : h2->GetYaxis();
  if (ax->GetXmin() > bins.front() or ax->GetXmax() < bins.back()) {
    edm::LogWarning("METTesterPostProcessor")
        << "Slices [" << bins.front() << ", " << bins.back() << "] outside the " << axis << " axis range ["
        << ax->GetXmin() << ", " << ax->GetXmax() << "] of " << src->getFullname() << ".";
    return ret;
  }

  for (unsigned binId = 0; binId < bins.size() - 1; binId++) {
    // the upper edge belongs to the next slice
    const int first = ax->FindFixBin(bins[binId]);
    int last = ax->FindFixBin(bins[binId + 1]);
    if (last > first and ax->GetBinLowEdge(last) >= bins[binId + 1]) {
      // remove the last bin if the bin value in `bins` matches the histogram bin edge, since that bin will be entirely outside the slice
      // but .ProjectionX/Y would sum it (leading to duplication), since it includes both `first` and `last`.
      --last;
    }

    const std::string name = src->getFullname() + "_proj" + std::to_string(binId);
    std::unique_ptr<TH1D> proj(axis == "X" ? h2->ProjectionY(name.c_str(), first, last)
                                           : h2->ProjectionX(name.c_str(), first, last));
    proj->SetDirectory(nullptr);
    ret.push_back({static_cast<float>(proj->GetMean()),
                   static_cast<float>(proj->GetMeanError()),
                   static_cast<float>(proj->GetRMS()),
                   static_cast<float>(proj->GetRMSError())});
  }
  return ret;
}

// mean (and RMS as error) of the X quantity in slices of the Y (generated) quantity
void METTesterPostProcessor::fillProjectionHisto(MElem *src, MElem *dest, const std::vector<float> &bins) {
  const auto stats = projectionMeanAndRMS(src, bins, "Y");
  for (unsigned binId = 0; binId < stats.size(); binId++) {
    dest->setBinContent(binId + 1, stats[binId].mean);
    dest->setBinError(binId + 1, stats[binId].rms);
  }
}

void METTesterPostProcessor::mFillAggrHistograms(const std::string &metdir,
                                                 const Inputs &in,
                                                 DQMStore::IBooker &ibook) {
  ibook.setCurrentFolder(metdir);

  // differences
  MElem *metDiffAggrMET = ibook.book1D("METDiffAggr_MET", "METDiffAggr_MET", mNMETBins, mMETBins.data());
  MElem *metDiffAggrPhi = ibook.book1D("METDiffAggr_Phi", "METDiffAggr_Phi", mNPhiBins, mPhiBins.data());
  MElem *metDeltaPhiAggrMET = ibook.book1D("METDeltaPhiAggr_MET", "METDeltaPhiAggr_MET", mNMETBins, mMETBins.data());
  MElem *metDeltaPhiAggrPhi = ibook.book1D("METDeltaPhiAggr_Phi", "METDeltaPhiAggr_Phi", mNPhiBins, mPhiBins.data());

  // response
  MElem *metRespAggrMET = ibook.book1D("METRespAggr_MET", "METRespAggr_MET", mNMETBins, mMETBins.data());
  MElem *metRespAggrPhi = ibook.book1D("METRespAggr_Phi", "METRespAggr_Phi", mNPhiBins, mPhiBins.data());

  // resolution
  MElem *metResolAggrPhi = ibook.book1D("METResolAggr_Phi", "METResolAggr_Phi", mNPhiBins, mPhiBins.data());
  MElem *metGenResolAggrPhi = ibook.book1D("METGenResolAggr_Phi", "METGenResolAggr_Phi", mNPhiBins, mPhiBins.data());
  MElem *metResolDiffAggrPhi = ibook.book1D("METResolDiffAggr_Phi", "METResolDiffAggr_Phi", mNPhiBins, mPhiBins.data());

  // significance
  MElem *metSignAggrPhi = ibook.book1D("METSignAggr_Phi", "METSignAggr_Phi", mNPhiBins, mPhiBins.data());
  MElem *metGenSignAggrPhi = ibook.book1D("METGenSignAggr_Phi", "METGenSignAggr_Phi", mNPhiBins, mPhiBins.data());
  MElem *metSignDiffAggrPhi = ibook.book1D("METSignDiffAggr_Phi", "METSignDiffAggr_Phi", mNPhiBins, mPhiBins.data());

  fillProjectionHisto(in.metDiff_vs_GenMETTrue, metDiffAggrMET, mMETBins);
  fillProjectionHisto(in.metDeltaPhi_vs_GenMETTrue, metDeltaPhiAggrMET, mMETBins);
  fillProjectionHisto(in.metDiff_vs_GenMETPhi, metDiffAggrPhi, mPhiBins);
  fillProjectionHisto(in.metDeltaPhi_vs_GenMETPhi, metDeltaPhiAggrPhi, mPhiBins);

  fillProjectionHisto(in.metRatio_vs_GenMETTrue, metRespAggrMET, mMETBins);
  fillProjectionHisto(in.metRatio_vs_GenMETPhi, metRespAggrPhi, mPhiBins);

  // reconstructed MET (Y axis) in slices of generated phi (X axis)
  const auto metRecoStats = projectionMeanAndRMS(in.genMETPhi_vs_MET, mPhiBins, "X");
  // generated MET (X axis) in slices of generated phi (Y axis)
  const auto metGenStats = projectionMeanAndRMS(in.genMETTrue_vs_GenMETPhi, mPhiBins, "Y");
  if (metRecoStats.size() != mNPhiBins or metGenStats.size() != mNPhiBins)
    return;

  for (unsigned binId = 0; binId < mNPhiBins; binId++) {
    const int bin = binId + 1;

    // reconstructed MET
    const float metMean = metRecoStats[binId].mean;
    const float metRMS = metRecoStats[binId].rms;
    const float metRMSErr = metRecoStats[binId].rmsErr;
    metResolAggrPhi->setBinContent(bin, metRMS);
    metResolAggrPhi->setBinError(bin, metRMSErr);

    const float significance = metRMS < mEpsilonFloat ? 0.f : metMean / metRMS;
    const float significanceErr = ((metRMS < mEpsilonFloat or metMean < mEpsilonFloat)
                                       ? 0.f
                                       : mComputeSignErr(significance, metRMS, metMean, metRMSErr));
    metSignAggrPhi->setBinContent(bin, significance);
    metSignAggrPhi->setBinError(bin, significanceErr);

    // generated MET
    const float metGenMean = metGenStats[binId].mean;
    const float metGenRMS = metGenStats[binId].rms;
    const float metGenRMSErr = metGenStats[binId].rmsErr;
    metGenResolAggrPhi->setBinContent(bin, metGenRMS);
    metGenResolAggrPhi->setBinError(bin, metGenRMSErr);

    const float significanceGen = metGenRMS < mEpsilonFloat ? 0.f : metGenMean / metGenRMS;
    const float significanceGenErr = ((metGenRMS < mEpsilonFloat or metGenMean < mEpsilonFloat)
                                          ? 0.f
                                          : mComputeSignErr(significanceGen, metGenRMS, metGenMean, metGenRMSErr));
    metGenSignAggrPhi->setBinContent(bin, significanceGen);
    metGenSignAggrPhi->setBinError(bin, significanceGenErr);

    // differences
    metResolDiffAggrPhi->setBinContent(bin, metRMS - metGenRMS);
    metResolDiffAggrPhi->setBinError(bin, std::sqrt(metGenRMSErr * metGenRMSErr + metRMSErr * metRMSErr));
    metSignDiffAggrPhi->setBinContent(bin, significance - significanceGen);
    metSignDiffAggrPhi->setBinError(
        bin, std::sqrt(significanceErr * significanceErr + significanceGenErr * significanceGenErr));
  }
}

// Compute significance error
float METTesterPostProcessor::mComputeSignErr(float significance, float metRMS, float metMean, float metRMSError) {
  return significance *
         std::sqrt((metRMS * metRMS / (metMean * metMean)) + (metRMSError * metRMSError / (metRMS * metRMS)));
}

void METTesterPostProcessor::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("runDir", "JetMET/METValidation/");
  descriptions.addWithDefaultLabel(desc);
}
