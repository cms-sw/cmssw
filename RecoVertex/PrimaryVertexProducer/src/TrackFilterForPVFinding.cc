#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
{
  maxD0Sig_    = conf.getParameter<double>("maxD0Significance");
  minPt_       = conf.getParameter<double>("minPt");
  maxNormChi2_ = conf.getParameter<double>("maxNormalizedChi2");
  minSiHits_   = conf.getParameter<int>("minSiliconHits");
  minPxHits_   = conf.getParameter<int>("minPixelHits");
  
}


bool
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
	if (!tk.stateAtBeamLine().isValid()) return false;
	bool IPSigCut = tk.stateAtBeamLine().transverseImpactParameter().significance()<maxD0Sig_;
	bool pTCut    = tk.impactPointState().globalMomentum().transverse() > minPt_;
	bool normChi2Cut  = tk.normalizedChi2() < maxNormChi2_;
	bool nSiHitsCut = tk.hitPattern().numberOfValidHits() > minSiHits_;
	bool nPxHitsCut = tk.hitPattern().numberOfValidPixelHits() > minPxHits_;

	return IPSigCut && pTCut && normChi2Cut && nSiHitsCut && nPxHitsCut;

	//return tk.stateAtBeamLine().transverseImpactParameter().significance()<maxD0Sig_;
}


float TrackFilterForPVFinding::minPt() const
{
  return minPt_;
}


float TrackFilterForPVFinding::maxD0Significance() const
{
  return maxD0Sig_;
}
