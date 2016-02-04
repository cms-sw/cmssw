#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
{
  maxD0Sig_    = conf.getParameter<double>("maxD0Significance");
  minPt_       = conf.getParameter<double>("minPt");
  maxNormChi2_ = conf.getParameter<double>("maxNormalizedChi2");
  minSiLayers_    = conf.getParameter<int>("minSiliconLayersWithHits");  
  minPxLayers_    = conf.getParameter<int>("minPixelLayersWithHits");  

  // the next few lines are taken from RecoBTag/SecondaryVertex/interface/TrackSelector.h"
  std::string qualityClass =
    conf.getParameter<std::string>("trackQuality");
  if (qualityClass == "any" || qualityClass == "Any" ||
      qualityClass == "ANY" || qualityClass == "") {
    quality_ = reco::TrackBase::undefQuality;
  } else {
    quality_ = reco::TrackBase::qualityByName(qualityClass);
  }

}

// select a single track
bool
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
	if (!tk.stateAtBeamLine().isValid()) return false;
	bool IPSigCut = tk.stateAtBeamLine().transverseImpactParameter().significance()<maxD0Sig_;
	bool pTCut    = tk.impactPointState().globalMomentum().transverse() > minPt_;
	bool normChi2Cut  = tk.normalizedChi2() < maxNormChi2_;
	bool nPxLayCut = tk.hitPattern().pixelLayersWithMeasurement() >= minPxLayers_;
	bool nSiLayCut =  tk.hitPattern().trackerLayersWithMeasurement() >= minSiLayers_;
	bool trackQualityCut = (quality_==reco::TrackBase::undefQuality)|| tk.track().quality(quality_);

	return IPSigCut && pTCut && normChi2Cut && nPxLayCut && nSiLayCut && trackQualityCut;
}



// select the vector of tracks that pass the filter cuts
std::vector<reco::TransientTrack> TrackFilterForPVFinding::select(const std::vector<reco::TransientTrack>& tracks) const
{
  std::vector <reco::TransientTrack> seltks;
  for (std::vector<reco::TransientTrack>::const_iterator itk = tracks.begin();
       itk != tracks.end(); itk++) {
    if ( operator()(*itk) ) seltks.push_back(*itk);  //  calls the filter function for single tracks
  }
  return seltks;
}
