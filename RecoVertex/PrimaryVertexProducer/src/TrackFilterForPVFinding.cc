#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf) {
  maxD0Sig_ = conf.getParameter<double>("maxD0Significance");
  maxD0Error_ = conf.getParameter<double>("maxD0Error");
  maxDzError_ = conf.getParameter<double>("maxDzError");
  minPt_ = conf.getParameter<double>("minPt");
  maxEta_ = conf.getParameter<double>("maxEta");
  maxNormChi2_ = conf.getParameter<double>("maxNormalizedChi2");
  minSiLayers_ = conf.getParameter<int>("minSiliconLayersWithHits");
  minPxLayers_ = conf.getParameter<int>("minPixelLayersWithHits");

  // the next few lines are taken from RecoBTag/SecondaryVertex/interface/TrackSelector.h"
  std::string qualityClass = conf.getParameter<std::string>("trackQuality");
  if (qualityClass == "any" || qualityClass == "Any" || qualityClass == "ANY" || qualityClass.empty()) {
    quality_ = reco::TrackBase::undefQuality;
  } else {
    quality_ = reco::TrackBase::qualityByName(qualityClass);
  }
}

// select a single track
bool TrackFilterForPVFinding::operator()(const reco::TransientTrack& tk) const {
  if (!tk.stateAtBeamLine().isValid())
    return false;
  bool IPSigCut = (tk.stateAtBeamLine().transverseImpactParameter().significance() < maxD0Sig_) &&
                  (tk.stateAtBeamLine().transverseImpactParameter().error() < maxD0Error_) &&
                  (tk.track().dzError() < maxDzError_);
  bool pTCut = tk.impactPointState().globalMomentum().transverse() > minPt_;
  bool etaCut = std::fabs(tk.impactPointState().globalMomentum().eta()) < maxEta_;
  bool normChi2Cut = tk.normalizedChi2() < maxNormChi2_;
  bool nPxLayCut = tk.hitPattern().pixelLayersWithMeasurement() >= minPxLayers_;
  bool nSiLayCut = tk.hitPattern().trackerLayersWithMeasurement() >= minSiLayers_;
  bool trackQualityCut = (quality_ == reco::TrackBase::undefQuality) || tk.track().quality(quality_);

  return IPSigCut && pTCut && etaCut && normChi2Cut && nPxLayCut && nSiLayCut && trackQualityCut;
}

// select the vector of tracks that pass the filter cuts
std::vector<reco::TransientTrack> TrackFilterForPVFinding::select(
    const std::vector<reco::TransientTrack>& tracks) const {
  std::vector<reco::TransientTrack> seltks;
  for (std::vector<reco::TransientTrack>::const_iterator itk = tracks.begin(); itk != tracks.end(); itk++) {
    if (operator()(*itk))
      seltks.push_back(*itk);  //  calls the filter function for single tracks
  }
  return seltks;
}

// select the vector of tracks that pass the filter cuts with a tighter pt selection
std::vector<reco::TransientTrack> TrackFilterForPVFinding::selectTight(const std::vector<reco::TransientTrack>& tracks,
                                                                       double minPtTight) const {
  std::vector<reco::TransientTrack> seltks;
  for (std::vector<reco::TransientTrack>::const_iterator itk = tracks.begin(); itk != tracks.end(); itk++) {
    if (itk->impactPointState().globalMomentum().transverse() < minPtTight)
      continue;
    if (operator()(*itk))
      seltks.push_back(*itk);  //  calls the filter function for single tracks
  }
  return seltks;
}

void TrackFilterForPVFinding::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<double>("maxNormalizedChi2", 10.0);
  desc.add<double>("minPt", 0.0);
  desc.add<std::string>("algorithm", "filter");
  desc.add<double>("maxEta", 2.4);
  desc.add<double>("maxD0Significance", 4.0);
  desc.add<double>("maxD0Error", 1.0);
  desc.add<double>("maxDzError", 1.0);
  desc.add<std::string>("trackQuality", "any");
  desc.add<int>("minPixelLayersWithHits", 2);
  desc.add<int>("minSiliconLayersWithHits", 5);
}
