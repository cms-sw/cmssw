#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
  : theConfig(conf) {}


bool
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  return tk.stateAtBeamLine().transverseImpactParameter().significance()<maxD0Significance();
}


float TrackFilterForPVFinding::minPt() const
{
  return theConfig.getParameter<double>("minPt");
}


float TrackFilterForPVFinding::maxD0Significance() const
{
  return theConfig.getParameter<double>("maxD0Significance");
}
