#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"

//#define DBG_MSU

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
  theDeltaCov(1,1) = 0.;
  theDeltaCov(1,2) = 0.;
  theDeltaCov(2,2) = 0.;
  //
  // Now get information on medium
  //
  if (surface.mediumProperties()) {
    // Momentum vector
    LocalVector d = TSoS.localMomentum();
    double p2 = d.mag2();
    d *= 1./sqrt(p2);
    // MediumProperties mp(0.02, .5e-4);
    const MediumProperties& mp = *surface.mediumProperties();
    double xf = 1./fabs(d.z());         // increase of path due to angle of incidence
    // calculate general physics things
    const double amscon = 1.8496e-4;    // (13.6MeV)**2
    const double m2 = mass()*mass();            // use mass hypothesis from constructor
    double e2     = p2 + m2;
    double beta2  = p2/e2;
    // calculate the multiple scattering angle
    double radLen = mp.radLen()*xf;     // effective rad. length
    double sigt2 = 0.;                  // sigma(alpha)**2
    if (radLen > 0) {
      // Calculated rms scattering angle squared.
      double fact = 1. + 0.038*log(radLen); fact *=fact;
      double a = fact/(beta2*p2);
      sigt2 = amscon*radLen*a;
      if (thePtMin > 0) {
#ifdef DBG_MSU
        std::cout<<"Original rms scattering = "<<sqrt(sigt2);
#endif
        // Inflate estimated rms scattering angle, to take into account 
        // that 1/p is not known precisely.
        AlgebraicSymMatrix55 const & covMatrix = TSoS.localError().matrix();
        double error2_QoverP = covMatrix(0,0);
	// Formula valid for ultra-relativistic particles.
//      sigt2 *= (1. + p2 * error2_QoverP);
	// Exact formula
        sigt2 *= (1. + p2 * error2_QoverP *
		               (1. + 5.*m2/e2 + 3.*m2*beta2*error2_QoverP));
#ifdef DBG_MSU
	std::cout<<" new = "<<sqrt(sigt2);
#endif
	// Convert Pt constraint to P constraint, neglecting uncertainty in 
	// track angle.
	double pMin2 = thePtMin*thePtMin*(p2/TSoS.globalMomentum().perp2());       
        // Use P constraint to calculate rms maximum scattering angle.
        //double betaMin2 = (pMin*pMin)/(pMin * pMin + m2);
        //double a_max = fact/(betaMin2 * pMin2);
        double a_max =  fact*(pMin2 + m2);
        double sigt2_max = amscon*radLen*a_max;
        if (sigt2 > sigt2_max) sigt2 = sigt2_max;
#ifdef DBG_MSU
	std::cout<<" after P constraint ("<<pMin<<") = "<<sqrt(sigt2);
	std::cout<<" for track with 1/p="<<1/p<<"+-"<<sqrt(error2_QoverP)<<std::endl;
#endif
      }
    }
    double sl2 = d.perp2();
    double cl2 = (d.z()*d.z());
    double cf2 = (d.x()*d.x())/sl2;
    double sf2 = (d.y()*d.y())/sl2;
    // Create update (transformation of independant variations
    //   on angle in orthogonal planes to local parameters.
    double den = 1./(cl2*cl2);
    theDeltaCov(1,1) = sigt2*(sf2*cl2 + cf2)*den;
    theDeltaCov(1,2) = sigt2*(cf2*sl2      )*den;
    theDeltaCov(2,2) = sigt2*(cf2*cl2 + sf2)*den;
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
  LocalVector localP = TSoS.localMomentum(); // let's call localMomentum only once
  return 
    localP.mag() != theLastP || 
    //TSoS.localMomentum().unit().z()!=theLastDz ||   // if we get there,  TSoS.localMomentum().mag() = theLastP!
    localP.z() != theLastDz*theLastP   ||   // so we can just do this, I think
    propDir!=theLastPropDir ||
    TSoS.surface().mediumProperties()->radLen()!=theLastRadLength;
}
//
// Save arguments
//
void MultipleScatteringUpdator::storeArguments (const TrajectoryStateOnSurface& TSoS, 
						const PropagationDirection propDir) const {
  LocalVector localP = TSoS.localMomentum(); // let's call localMomentum only once
  theLastP = localP.mag();
  theLastDz = (theLastP == 0 ? 0 : localP.z()/theLastP);
  theLastPropDir = propDir;
  theLastRadLength = TSoS.surface().mediumProperties()->radLen();
}

