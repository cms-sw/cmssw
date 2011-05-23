#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

GlobalTrajectoryParameters::GlobalTrajectoryParameters(const GlobalPoint& aX,
						       const GlobalVector& direction,
						       double transverseCurvature, int, 
						       const MagneticField* fieldProvider) :
  theX(aX), theField(fieldProvider), hasCurvature_(true), cachedCurvature_(transverseCurvature)
{
  double bza = -2.99792458e-3 * theField->inTesla(theX).z();
  double qbp = transverseCurvature/bza*direction.perp();
  theP = direction*fabs(1./qbp);
  theCharge = qbp > 0. ? 1 : -1;
}

double GlobalTrajectoryParameters::transverseCurvature() const
{
  if (!hasCurvature_) {
      double bza = -2.99792458e-3 * theField->inTesla(theX).z();
      cachedCurvature_ = bza*signedInverseTransverseMomentum();
      hasCurvature_ = true;
  }
  return cachedCurvature_;
}

GlobalVector GlobalTrajectoryParameters::magneticFieldInInverseGeV( const GlobalPoint& x) const
{
  return 2.99792458e-3 * theField->inTesla(x);
}
