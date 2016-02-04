#include "TrackingTools/MaterialEffects/interface/VolumeMultipleScatteringEstimator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsEstimator.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsEstimate.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMediumProperties.h"

VolumeMaterialEffectsEstimate 
VolumeMultipleScatteringEstimator::estimate (const TrajectoryStateOnSurface refTSOS,
					     double pathLength,
					     const VolumeMediumProperties& medium) const
{
  //
  // Initialise the update to the covariance matrix
  // (dP is constantly 0).
  //
  AlgebraicSymMatrix55 deltaCov;        // assume initialization to 0 ...
  //
  // Now get information on medium
  //
  // Momentum vector
  double p = refTSOS.localMomentum().mag();
  // MediumProperties mp(0.02, .5e-4);
  // calculate general physics things
  const double amscon = 1.8496e-4;    // (13.6MeV)**2
  const double m = mass();            // use mass hypothesis from constructor
  double e     = sqrt(p*p + m*m);
  double beta  = p/e;
  // calculate the multiple scattering angle
  double radLen = pathLength / medium.x0(); // effective rad. length
  double sigth2 = 0.;                       // sigma(theta)
  if (radLen > 0) {
    double a = (1. + 0.038*log(radLen))/(beta*p);
    sigth2 = amscon*radLen*a*a;
  }
  // Create update (transformation of independant variations
  //   on positions and angles in a cartesian system 
  //   with z-axis parallel to the track.
  deltaCov(1,1) = deltaCov(2,2) = sigth2;
  deltaCov(3,3) = deltaCov(4,4) = sigth2/3.*pathLength*pathLength;
  deltaCov(1,3) = deltaCov(3,1) = 
    deltaCov(2,4) = deltaCov(4,2) = sigth2/2.;     // correlation of sqrt(3)/2
  return VolumeMaterialEffectsEstimate(0.,deltaCov);
}

VolumeMultipleScatteringEstimator*
VolumeMultipleScatteringEstimator::clone () const
{
  return new VolumeMultipleScatteringEstimator(*this);
}
