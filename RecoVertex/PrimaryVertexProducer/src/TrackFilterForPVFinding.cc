#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
  : theConfig(conf) {}


bool 
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  /*
  PerigeeTrajectoryParameters::ParameterVector  p = tk.parameters();
  PerigeeTrajectoryError::CovarianceMatrix c = tk.covariance();
 double d0Error=sqrt(c(3,3));
  */
  return ( (tk.initialFreeState().momentum().perp() > minPt())
	  && (std::abs(tk.impactPointTSCP().perigeeParameters().transverseImpactParameter() 
	  	/ tk.impactPointTSCP().perigeeError().transverseImpactParameterError()) < maxD0Significance()));
}


float TrackFilterForPVFinding::minPt() const 
{
  return theConfig.getParameter<double>("minPt");
}


float TrackFilterForPVFinding::maxD0Significance() const 
{
  return theConfig.getParameter<double>("maxD0Significance");
}
