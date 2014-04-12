#include "TrackingTools/GsfTracking/interface/GsfMultipleScatteringUpdator.h"

#include "DataFormats/GeometrySurface/interface/MediumProperties.h"

// #include "CommonDet/DetUtilities/interface/DetExceptions.h"

// #include "Utilities/Notification/interface/Verbose.h"

void
GsfMultipleScatteringUpdator::compute (const TrajectoryStateOnSurface& TSoS,
				       const PropagationDirection propDir, Effect effects[]) const 
{
  //
  // Get surface and check presence of medium properties
  //
  const Surface& surface = TSoS.surface();
  //
  // calculate components
  //
  if ( surface.mediumProperties().isValid() ) {
    LocalVector pvec = TSoS.localMomentum();
    float p = TSoS.localMomentum().mag();
    pvec *= 1./p;
    // thickness in radiation lengths
    float rl = surface.mediumProperties().radLen()/fabs(pvec.z());
    // auxiliary variables for modified X0
    constexpr float z = 14;                 // atomic number of silicon
    const float logz = log(z);
    const float h = (z+1)/z*log(287*sqrt(z))/log(159*pow(z,-1./3.));
    float beta2 = 1./(1.+mass()*mass()/p/p);
    // reduced thickness
    float dp1 = rl/beta2/h;
    float logdp1 = log(dp1);
    float logdp2 = 2./3.*logz + logdp1;
    // weights
    float w2;
    if ( logdp2<log(0.5) )  
      w2 = 0.05283+0.0077*logdp2+0.00069*logdp2*logdp2;
    else
      w2 =-0.01517+0.1151*logdp2-0.00653*logdp2*logdp2;
    float w1 = 1.-w2;
    effects[0].weight*=w1;
    effects[1].weight*=w2;
    // reduced variances
    float var1 = 0.8510+0.03314*logdp1-0.001825*logdp1*logdp1;
    float var2 = (1.-w1*var1)/w2;
    for ( int ic=0; ic<2; ic++ ) {
      // choose component and multiply with total variance
      float var = ic==0 ? var1 : var2;
      var *= 225.e-6*dp1/p/p;
      AlgebraicSymMatrix55 cov;
      // transform from orthogonal planes containing the
      // momentum vector to local parameters
      float sl = pvec.perp();
      float cl = pvec.z();
      float cf = pvec.x()/sl;
      float sf = pvec.y()/sl;
      using namespace materialEffect;
      effects[ic].deltaCov[msxx] += var*(sf*sf*cl*cl + cf*cf)/(cl*cl*cl*cl);
      effects[ic].deltaCov[msxy] += var*(cf*sf*sl*sl        )/(cl*cl*cl*cl);
      effects[ic].deltaCov[msyy] += var*(cf*cf*cl*cl + sf*sf)/(cl*cl*cl*cl);
    }
  }

}

