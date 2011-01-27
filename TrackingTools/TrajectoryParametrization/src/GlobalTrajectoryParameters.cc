#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

GlobalTrajectoryParameters::GlobalTrajectoryParameters(const GlobalPoint& aX,
						       const GlobalVector& direction,
						       float transverseCurvature, int, 
						       const MagneticField* fieldProvider) :
  theField(fieldProvider),
  theX(aX), cachedCurvature_(transverseCurvature),  hasCurvature_(true)
{
  float bza = -2.99792458e-3f * theField->inTesla(theX).z();
  float qbpi = bza/(direction.perp()*transverseCurvature);
  theP = direction*std::abs(qbpi);
  theCharge = qbpi > 0. ? 1 : -1;
}

float GlobalTrajectoryParameters::transverseCurvature() const
{
  if (!hasCurvature_) {
      float bza = -2.99792458e-3f * theField->inTesla(theX).z();
      cachedCurvature_ = bza*signedInverseTransverseMomentum();
      hasCurvature_ = true;
  }
  return cachedCurvature_;
}

GlobalVector GlobalTrajectoryParameters::magneticFieldInInverseGeV( const GlobalPoint& x) const
{
  return 2.99792458e-3f * theField->inTesla(x);
}


/*  the field is different as it is attached to each given volume!!!!
// const MagneticField* GlobalTrajectoryParameters::theField=0;
#include<iostream>
// FIXME debug code mostly
void GlobalTrajectoryParameters::setMF(const MagneticField* fieldProvider) {
  if (0==fieldProvider) return;
  if (0!=theField && fieldProvider!=theField)
    std::cout << "GlobalTrajectoryParameters: a different MF???? " 
	      << theField << " " << fieldProvider << std::endl;
  theField =fieldProvider;
}
*/
