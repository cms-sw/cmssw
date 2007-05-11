#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
  : theConfig(conf) {}


bool 
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  // obsolete as of 1_5_0_pre3
  double d0=tk.impactPointTSCP().position().perp();
  double s0=sqrt(tk.impactPointTSCP().perigeeError().covarianceMatrix()(3,3)); 
  return d0<s0*maxD0Significance();
  
}

bool 
TrackFilterForPVFinding::operator() (const BeamTransientTrack & tk) const
{
  return tk.impactParameterSignificance()<maxD0Significance();
}


float TrackFilterForPVFinding::minPt() const 
{
  return theConfig.getParameter<double>("minPt");
}


float TrackFilterForPVFinding::maxD0Significance() const 
{
  return theConfig.getParameter<double>("maxD0Significance");
}
