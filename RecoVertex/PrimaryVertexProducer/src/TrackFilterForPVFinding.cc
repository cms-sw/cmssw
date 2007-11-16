#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
{
  maxD0Sig_ = conf.getParameter<double>("maxD0Significance");
  minPt_ = conf.getParameter<double>("minPt");
  }


bool
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  return tk.stateAtBeamLine().transverseImpactParameter().significance()<maxD0Sig_;
}


float TrackFilterForPVFinding::minPt() const
{
  return minPt_;
}


float TrackFilterForPVFinding::maxD0Significance() const
{
  return maxD0Sig_;
}
