#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include <cmath>

PerigeeTrajectoryParameters PerigeeConversions::ftsToPerigeeParameters
  (const FTS& originalFTS, const GlobalPoint& referencePoint) const

{
  GlobalVector impactDistance = originalFTS.position() - referencePoint;

  double theta = originalFTS.momentum().theta();
  double phi = originalFTS.momentum().phi();
  double pt = originalFTS.momentum().perp();
//   double field = MagneticField::inInverseGeV(originalFTS.position()).z();
  double field  = originalFTS.parameters().magneticField().inTesla(originalFTS.position()).z() * 2.99792458e-3;

  double positiveMomentumPhi = ( (phi>0) ? phi : (2*M_PI+phi) );
  double positionPhi = impactDistance.phi();
  double positivePositionPhi =
   ( (positionPhi>0) ? positionPhi : (2*M_PI+positionPhi) );
  double phiDiff = positiveMomentumPhi - positivePositionPhi;
  if (phiDiff<0.0) phiDiff+= (2*M_PI);
  double signEpsilon = ( (phiDiff > M_PI) ? -1.0 : 1.0);

  double epsilon = signEpsilon *
  		     sqrt ( impactDistance.x()*impactDistance.x() +
   			    impactDistance.y()*impactDistance.y() );

  // The track parameters:
  AlgebraicVector theTrackParameters = AlgebraicVector(5);

  double signTC = -originalFTS.charge();
  bool isCharged = (signTC!=0);
  if (isCharged) {
    theTrackParameters[0] = field / pt*signTC;
  } else {
    theTrackParameters[0] = 1 / pt;
    
  }
  theTrackParameters[1] = theta;
  theTrackParameters[2] = phi;
  theTrackParameters[3] = epsilon;
  theTrackParameters[4] = impactDistance.z();
  return PerigeeTrajectoryParameters(theTrackParameters, isCharged);
}


GlobalPoint PerigeeConversions::positionFromPerigee
  (const PerigeeTrajectoryParameters& parameters, const GlobalPoint& referencePoint) const
{
  AlgebraicVector theVector = parameters.vector();
  return GlobalPoint(theVector[3]*sin(theVector[2])+referencePoint.x(),
  		     -theVector[3]*cos(theVector[2])+referencePoint.y(),
		     theVector[4]+referencePoint.z());
}


GlobalVector PerigeeConversions::momentumFromPerigee
  (const PerigeeTrajectoryParameters& parameters, const GlobalPoint& referencePoint,
  const MagneticField& magField) const
{
  return momentumFromPerigee(parameters.vector(), parameters.charge(), referencePoint,
  				magField);
}

GlobalVector PerigeeConversions::momentumFromPerigee
  (const AlgebraicVector& momentum, const TrackCharge& charge, 
   const GlobalPoint& referencePoint, const MagneticField& magField) const
{
  double pt;
  if (charge!=0) {
//     pt = abs(MagneticField::inInverseGeV(referencePoint).z() / momentum[0]);
    pt = std::abs(magField.inTesla(referencePoint).z() * 2.99792458e-3 / momentum[0]);
  } else {
    pt = 1 / momentum[0];
  }
  return GlobalVector(cos(momentum[2]) * pt,
  		      sin(momentum[2]) * pt,
   		      pt/tan(momentum[1]));
}

TrackCharge PerigeeConversions::chargeFromPerigee
  (const PerigeeTrajectoryParameters& parameters, const GlobalPoint& referencePoint) const
{
  return parameters.charge();
}


TrajectoryStateClosestToPoint PerigeeConversions::trajectoryStateClosestToPoint
	(const AlgebraicVector& momentum, const GlobalPoint& referencePoint,
	 const TrackCharge& charge, const AlgebraicMatrix& theCovarianceMatrix,
	 const MagneticField& magField) const
{
  AlgebraicMatrix param2cart = jacobianParameters2Cartesian
  	(momentum, referencePoint, charge, magField);
  AlgebraicSymMatrix cartesianErrorMatrix(6,0);
  cartesianErrorMatrix.assign(param2cart*theCovarianceMatrix*param2cart.T());
  CartesianTrajectoryError cartesianTrajErr(cartesianErrorMatrix);

  FTS theFTS(GlobalTrajectoryParameters(referencePoint,
	    momentumFromPerigee(momentum, charge, referencePoint, magField), charge, &magField),
	    cartesianTrajErr);

  return TrajectoryStateClosestToPoint(theFTS, referencePoint);
}


AlgebraicMatrix
PerigeeConversions::jacobianParameters2Cartesian(
	const AlgebraicVector& momentum, const GlobalPoint& position,
	const TrackCharge& charge, const MagneticField& magField) const
{
  double factor = 1.;
  if (charge!=0) {
    double field = magField.inTesla(position).z() * 2.99792458e-3;
//     MagneticField::inInverseGeV(position).z();
    factor = -field*charge;
  }
  AlgebraicMatrix frameTransJ(6, 6, 0);
  frameTransJ[0][0] = 1;
  frameTransJ[1][1] = 1;
  frameTransJ[2][2] = 1;
  frameTransJ[3][3] = - factor * cos(momentum[2]) / (momentum[0]*momentum[0]);
  frameTransJ[4][3] = - factor * sin(momentum[2]) / (momentum[0]*momentum[0]);
  frameTransJ[5][3] = - factor / tan(momentum[1]) / (momentum[0]*momentum[0]);

  frameTransJ[3][5] = - factor * sin(momentum[2])  / (momentum[0]);
  frameTransJ[4][5] = factor * cos(momentum[2]) / (momentum[0]);
  frameTransJ[5][4] = - factor / (momentum[0]*sin(momentum[1])*sin(momentum[1]));

  return frameTransJ;
}
