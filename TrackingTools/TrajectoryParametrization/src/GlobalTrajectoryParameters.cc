#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/Utilities/interface/Likely.h"

GlobalTrajectoryParameters::GlobalTrajectoryParameters(const GlobalPoint& aX,
                                                       const GlobalVector& direction,
                                                       float transverseCurvature,
                                                       int,
                                                       const MagneticField* fieldProvider)
    : theField(fieldProvider), theX(aX) {
  cachedMagneticField = theField->inTesla(theX);
  float bza = -2.99792458e-3f * cachedMagneticField.z();
  float qbpi = bza / (direction.perp() * transverseCurvature);
  theP = direction * std::abs(qbpi);
  theCharge = qbpi > 0.f ? 1 : -1;
}

GlobalTrajectoryParameters::GlobalTrajectoryParameters(const GlobalPoint& aX,
                                                       const GlobalVector& direction,
                                                       float transverseCurvature,
                                                       int,
                                                       const MagneticField* fieldProvider,
                                                       GlobalVector fieldValue)
    : theField(fieldProvider), theX(aX), cachedMagneticField(fieldValue) {
  float bza = -2.99792458e-3f * cachedMagneticField.z();
  float qbpi = bza / (direction.perp() * transverseCurvature);
  theP = direction * std::abs(qbpi);
  theCharge = qbpi > 0.f ? 1 : -1;
}

void GlobalTrajectoryParameters::setCache() {
  cachedMagneticField = theField ? theField->inTesla(theX) : GlobalVector(0, 0, 0);
}  // we must initialize cache to non-NAN to avoid FPE

GlobalVector GlobalTrajectoryParameters::magneticFieldInInverseGeV(const GlobalPoint& x) const {
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
