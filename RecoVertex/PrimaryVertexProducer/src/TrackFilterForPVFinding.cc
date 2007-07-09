#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include <cmath>

TrackFilterForPVFinding::TrackFilterForPVFinding(const edm::ParameterSet& conf)
  : theConfig(conf) {}


bool 
TrackFilterForPVFinding::operator() (const reco::TransientTrack & tk) const
{
  // return tk.stateAtBeamLine().transverseImpactParameter().significance()<maxD0Significance();
  // the above doesn't seem to work
  // temporary fix:
  AlgebraicSymMatrix33 error = tk.stateAtBeamLine().beamSpot().covariance3D() +
	tk.stateAtBeamLine().trackStateAtPCA().cartesianError().matrix().Sub<AlgebraicSymMatrix33>(0,0);
  GlobalPoint impactPoint=tk.stateAtBeamLine().trackStateAtPCA().position();
  double dx=impactPoint.x()-tk.stateAtBeamLine().beamLinePCA().x();
  double dy=impactPoint.y()-tk.stateAtBeamLine().beamLinePCA().y();
  AlgebraicVector3 transverseFlightPath(dx,dy,0.);
  double ip      = sqrt(dx*dx+dy*dy);
  double ipError = sqrt( ROOT::Math::Similarity(transverseFlightPath.Unit(),error) );
  //end of temporary fix
   return ip<ipError*maxD0Significance();
}


float TrackFilterForPVFinding::minPt() const 
{
  return theConfig.getParameter<double>("minPt");
}


float TrackFilterForPVFinding::maxD0Significance() const 
{
  return theConfig.getParameter<double>("maxD0Significance");
}
