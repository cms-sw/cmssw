#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

GlobalTrajectoryParameters::GlobalTrajectoryParameters(const GlobalPoint& aX,
						       const GlobalVector& direction,
						       double transverseCurvature, int, 
						       const MagneticField* fieldProvider) :
  theX(aX), theField(fieldProvider)
{
  double bza = -2.99792458e-3 * theField->inTesla(theX).z();
  double qbp = transverseCurvature/bza*direction.perp();
  theP = direction*fabs(1./qbp);
  theCharge = qbp > 0. ? 1 : -1;
}

double GlobalTrajectoryParameters::transverseCurvature() const
{
  double bza = -2.99792458e-3 * theField->inTesla(theX).z();
  return bza*signedInverseTransverseMomentum();
}
