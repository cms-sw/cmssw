#include "TrackingTools/PatternTools/interface/MediumProperties.h"
#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"

//
// Computation of contribution of multiple scatterning to covariance matrix 
//   of local parameters based on Highland formula for sigma(alpha) in plane.
//
void MultipleScatteringUpdator::compute (const TrajectoryStateOnSurface& TSoS, 
					 const PropagationDirection propDir) const
{
  //
  // Get surface
  //
  const Surface& surface = TSoS.surface();
  //
  // Initialise the update to the covariance matrix
  // (dP is constantly 0).
  //
  theDeltaCov(2,2) = 0.;
  theDeltaCov(2,3) = 0.;
  theDeltaCov(3,3) = 0.;
  //
  // Now get information on medium
  //
  if (surface.mediumProperties()) {
    // Momentum vector
    LocalVector d = TSoS.localMomentum();
    double p = d.mag();
    d *= 1./p;
    // MediumProperties mp(0.02, .5e-4);
    const MediumProperties& mp = *surface.mediumProperties();
    double xf = 1./fabs(d.z());         // increase of path due to angle of incidence
    // calculate general physics things
    const double amscon = 1.8496e-4;    // (13.6MeV)**2
    const double m = mass();            // use mass hypothesis from constructor
    double e     = sqrt(p*p + m*m);
    double beta  = p/e;
    // calculate the multiple scattering angle
    double radLen = mp.radLen()*xf;     // effective rad. length
    double sigt2 = 0.;                  // sigma(alpha)**2
    if (radLen > 0) {
      double a = (1. + 0.038*log(radLen))/(beta*p); a *= a;
      sigt2 = amscon*radLen*a;
    }
    double sl = d.perp();
    double cl = d.z();
    double cf = d.x()/sl;
    double sf = d.y()/sl;
    // Create update (transformation of independant variations
    //   on angle in orthogonal planes to local parameters.
    theDeltaCov(2,2) = sigt2*(sf*sf*cl*cl + cf*cf)/(cl*cl*cl*cl);
    theDeltaCov(2,3) = sigt2*(cf*sf*sl*sl        )/(cl*cl*cl*cl);
    theDeltaCov(3,3) = sigt2*(cf*cf*cl*cl + sf*sf)/(cl*cl*cl*cl);
  }
  //
  // Save arguments to avoid duplication of computation
  //
  storeArguments(TSoS,propDir);
}
//
// Compare arguments with the ones of the previous call
//
bool MultipleScatteringUpdator::newArguments (const TrajectoryStateOnSurface& TSoS, 
					      const PropagationDirection propDir) const {
  return TSoS.localMomentum().unit().z()!=theLastDz ||
    TSoS.localMomentum().mag()!=theLastP || propDir!=theLastPropDir ||
    TSoS.surface().mediumProperties()->radLen()!=theLastRadLength;
}
//
// Save arguments
//
void MultipleScatteringUpdator::storeArguments (const TrajectoryStateOnSurface& TSoS, 
						const PropagationDirection propDir) const {
  theLastDz = TSoS.localMomentum().unit().z();
  theLastP = TSoS.localMomentum().mag();
  theLastPropDir = propDir;
  theLastRadLength = TSoS.surface().mediumProperties()->radLen();
}

