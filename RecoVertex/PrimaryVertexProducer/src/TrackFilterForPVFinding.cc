#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
{
  maxD0Sig_    = conf.getParameter<double>("maxD0Significance");
  minPt_       = conf.getParameter<double>("minPt");
  maxNormChi2_ = conf.getParameter<double>("maxNormalizedChi2");
  minSiHits_   = conf.getParameter<int>("minSiliconHits"); // deprecated
  minPxHits_   = conf.getParameter<int>("minPixelHits");   // deprecated
  minSiLayers_    = conf.getParameter<int>("minSiliconLayersWithHits");  
  minPxLayers_    = conf.getParameter<int>("minPixelLayersWithHits");  

  // the next few lines are taken from RecoBTag/SecondaryVertex/interface/TrackSelector.h"
//   std::string qualityClass =
//     conf.getParameter<std::string>("trackQuality");
//   if (qualityClass == "any" || qualityClass == "Any" ||
//       qualityClass == "ANY" || qualityClass == "") {
//     quality_ = reco::TrackBase::undefQuality;
//   } else {
//     quality_ = reco::TrackBase::qualityByName(qualityClass);
//   }
  
//   if ((minSiHits_>-1)||( minPxHits_>-1)){   edm::LogInfo("RecoVertex/TrackFilterForPVFinding") 
//     << "You are using an obsolete selection, pleas use minSiliconLayers and minPixelLayers instead " << "\n";
//   }
}


bool
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
	if (!tk.stateAtBeamLine().isValid()) return false;
	bool IPSigCut = tk.stateAtBeamLine().transverseImpactParameter().significance()<maxD0Sig_;
	bool pTCut    = tk.impactPointState().globalMomentum().transverse() > minPt_;
	bool normChi2Cut  = tk.normalizedChi2() < maxNormChi2_;
	bool nSiHitsCut = tk.hitPattern().numberOfValidHits() > minSiHits_;  // deprecated
	bool nPxHitsCut = tk.hitPattern().numberOfValidPixelHits() >= minPxHits_;  //deprecated
	bool nPxLayCut = tk.hitPattern().pixelLayersWithMeasurement() >= minPxLayers_;
	bool nSiLayCut =  tk.hitPattern().trackerLayersWithMeasurement() >= minSiLayers_;
	//bool trackQualityCut = tk.track().quality(quality_); 
	return IPSigCut && pTCut && normChi2Cut && nSiHitsCut && nPxHitsCut && nPxLayCut && nSiLayCut ;
}


float TrackFilterForPVFinding::minPt() const
{
  return minPt_;
}


float TrackFilterForPVFinding::maxD0Significance() const
{
  return maxD0Sig_;
}
