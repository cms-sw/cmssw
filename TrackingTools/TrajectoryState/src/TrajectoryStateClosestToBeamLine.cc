#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"

TrajectoryStateClosestToBeamLine::TrajectoryStateClosestToBeamLine
	(const FreeTrajectoryState& stateAtPCA, const GlobalPoint & pointOnBeamLine,
	 const reco::BeamSpot& beamSpot) :
	  valid(true), theFTS(stateAtPCA) , thePointOnBeamLine(pointOnBeamLine),
	theBeamSpot(beamSpot)
{}

Measurement1D TrajectoryStateClosestToBeamLine::transverseImpactParameter() const
{
  if (!isValid()) throw TrajectoryStateException(
    "TrajectoryStateClosestToBeamLine is invalid.");
  AlgebraicSymMatrix33 error = theBeamSpot.rotatedCovariance3D() +
	theFTS.cartesianError().matrix().Sub<AlgebraicSymMatrix33>(0,0);

  GlobalPoint impactPoint=theFTS.position();
  AlgebraicVector3 transverseFlightPath(
	impactPoint.x()-thePointOnBeamLine.x(),impactPoint.y()-thePointOnBeamLine.y(),0.);
  double length = ROOT::Math::Mag(transverseFlightPath);
  // Warning: after the transverseFlightPath.Unit() statement, the
  // transverseFlightPath vector is CHANGED to a UNIT vector.
  double ipError = sqrt( ROOT::Math::Similarity(transverseFlightPath.Unit(),error) );
  return Measurement1D (length, ipError);
}

