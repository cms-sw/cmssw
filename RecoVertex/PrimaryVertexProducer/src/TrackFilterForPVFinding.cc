#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
  : theConfig(conf) {}


bool 
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  double d0=tk.impactPointTSCP().position().perp();
  double s0=sqrt(tk.impactPointTSCP().perigeeError().covarianceMatrix()(4,4)); 
  // note: switch to tk.impactPointTSCP().perigeeError().transverseImpactParameter when TransientTrack is fixed
  return d0<s0*maxD0Significance();
  
}


float TrackFilterForPVFinding::minPt() const 
{
  return theConfig.getParameter<double>("minPt");
}


float TrackFilterForPVFinding::maxD0Significance() const 
{
  return theConfig.getParameter<double>("maxD0Significance");
}
