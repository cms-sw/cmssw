#include "TrackingTools/GsfTracking/interface/GsfMultipleScatteringUpdator.h"

#include "DataFormats/GeometrySurface/interface/MediumProperties.h"

// #include "CommonDet/DetUtilities/interface/DetExceptions.h"

// #include "Utilities/Notification/interface/Verbose.h"

void
GsfMultipleScatteringUpdator::compute (const TrajectoryStateOnSurface& TSoS,
				       const PropagationDirection propDir) const 
{
  //
  // clear cache
  //
  theWeights.clear();
  theDeltaPs.clear();
  theDeltaCovs.clear();
  //
  // Get surface and check presence of medium properties
  //
  const Surface& surface = TSoS.surface();
  //
  // calculate components
  //
  if ( surface.mediumProperties() ) {
    LocalVector pvec = TSoS.localMomentum();
    double p = TSoS.localMomentum().mag();
    pvec *= 1./p;
    // thickness in radiation lengths
    double rl = surface.mediumProperties()->radLen()/fabs(pvec.z());
    // auxiliary variables for modified X0
    const double z = 14;                 // atomic number of silicon
    const double logz = log(z);
    const double h = (z+1)/z*log(287*sqrt(z))/log(159*pow(z,-1./3.));
    double beta2 = 1./(1.+mass()*mass()/p/p);
    // reduced thickness
    double dp1 = rl/beta2/h;
    double logdp1 = log(dp1);
    double logdp2 = 2./3.*logz + logdp1;
    // means are always 0
    theDeltaPs.push_back(0.);
    theDeltaPs.push_back(0.);
    // weights
    double w2;
    if ( logdp2<log(0.5) )  
      w2 = 0.05283+0.0077*logdp2+0.00069*logdp2*logdp2;
    else
      w2 =-0.01517+0.1151*logdp2-0.00653*logdp2*logdp2;
    double w1 = 1.-w2;
    theWeights.push_back(w1);
    theWeights.push_back(w2);
    // reduced variances
    double var1 = 0.8510+0.03314*logdp1-0.001825*logdp1*logdp1;
    double var2 = (1.-w1*var1)/w2;
    for ( int ic=0; ic<2; ic++ ) {
      // choose component and multiply with total variance
      double var = ic==0 ? var1 : var2;
      var *= 225.e-6*dp1/p/p;
      AlgebraicSymMatrix55 cov;
      // transform from orthogonal planes containing the
      // momentum vector to local parameters
      double sl = pvec.perp();
      double cl = pvec.z();
      double cf = pvec.x()/sl;
      double sf = pvec.y()/sl;
      cov(1,1) = var*(sf*sf*cl*cl + cf*cf)/(cl*cl*cl*cl);
      cov(1,2) = var*(cf*sf*sl*sl        )/(cl*cl*cl*cl);
      cov(2,2) = var*(cf*cf*cl*cl + sf*sf)/(cl*cl*cl*cl);
      theDeltaCovs.push_back(cov);
    }
  }
  else {
    theWeights.push_back(1.);
    theDeltaPs.push_back(0.);
    theDeltaCovs.push_back(AlgebraicSymMatrix55());
  }
  //
  // Save arguments to avoid duplication of computation
  //
  storeArguments(TSoS,propDir); 
}

//
// Compare arguments with the ones of the previous call
//
bool 
GsfMultipleScatteringUpdator::newArguments (const TrajectoryStateOnSurface& TSoS, 
					    const PropagationDirection propDir) const {
  return TSoS.localMomentum().unit().z()!=theLastDz ||
    TSoS.localMomentum().mag()!=theLastP || propDir!=theLastPropDir ||
    TSoS.surface().mediumProperties()->radLen()!=theLastRadLength;
}
//
// Save arguments
//
void GsfMultipleScatteringUpdator::storeArguments (const TrajectoryStateOnSurface& TSoS, 
						   const PropagationDirection propDir) const {
  theLastDz = TSoS.localMomentum().unit().z();
  theLastP = TSoS.localMomentum().mag();
  theLastPropDir = propDir;
  theLastRadLength = TSoS.surface().mediumProperties()->radLen();
}
