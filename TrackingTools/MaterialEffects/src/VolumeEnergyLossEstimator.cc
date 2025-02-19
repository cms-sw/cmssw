#include "TrackingTools/MaterialEffects/interface/VolumeEnergyLossEstimator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsEstimator.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsEstimate.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMediumProperties.h"

VolumeMaterialEffectsEstimate 
VolumeEnergyLossEstimator::estimate (const TrajectoryStateOnSurface refTSOS,
				     double pathLength,
				     const VolumeMediumProperties& medium) const
{
  //
  // Initialise dP and the update to the covariance matrix
  //
  double deltaP = 0.;
  double deltaCov00 = 0.;
  //
  // Bethe-Bloch
  //
  if ( mass()>0.001 )
    computeBetheBloch(refTSOS.localMomentum(),pathLength,medium,
		      deltaP,deltaCov00);
  //
  // Special treatment for electrons (currently rather crude
  // distinction using mass)
  //
  else
    computeElectrons(refTSOS.localMomentum(),pathLength,medium,
		     deltaP,deltaCov00);

  AlgebraicSymMatrix55 deltaCov;
  deltaCov(0,0) = deltaCov00;
  return VolumeMaterialEffectsEstimate(deltaP,deltaCov);
}


VolumeEnergyLossEstimator*
VolumeEnergyLossEstimator::clone () const
{
  return new VolumeEnergyLossEstimator(*this);
}

//
// Computation of energy loss according to Bethe-Bloch
//
void
VolumeEnergyLossEstimator::computeBetheBloch (const LocalVector& localP, double pathLength,
					    const VolumeMediumProperties& medium,
					    double& deltaP, double& deltaCov00) const {
  //
  // calculate absolute momentum and correction to path length from angle 
  // of incidence
  //
  double p = localP.mag();
  // constants
  const double m = mass();            // use mass hypothesis from constructor
  const double emass = 0.511e-3;
  // FIXME: replace constants for Si
  const double poti = 16.e-9 * 10.75; // = 16 eV * Z**0.9, for Si Z=14
  const double eplasma = 28.816e-9 * sqrt(2.33*0.498); // 28.816 eV * sqrt(rho*(Z/A)) for Si
  const double delta0 = 2*log(eplasma/poti) - 1.;
  // calculate general physics things
  double e     = sqrt(p*p + m*m);
  double beta  = p/e;
  double gamma = e/m;
  double eta2  = beta*gamma; eta2 *= eta2;
  double lnEta2 = log(eta2);
  double ratio = emass/m;
  double emax  = 2.*emass*eta2/(1. + 2.*ratio*gamma + ratio*ratio);
  double delta = delta0 + lnEta2;
  // calculate the mean and sigma of energy loss
  // xi = d[g/cm2] * 0.307075MeV/(g/cm2) * Z/A * 1/2
  double xi = pathLength * medium.xi(); xi /= (beta*beta);
//   double dEdx = xi*(log(2.*emass*eta2*emax/(poti*poti)) - 2.*(beta*beta));
  double dEdx = xi*(log(2.*emass*emax/(poti*poti))+lnEta2 - 2.*(beta*beta) - delta);
  double dEdx2 = xi*emax*(1.-beta*beta/2.);
  double dP    = dEdx/beta;
  double sigp2 = dEdx2*e*e/(p*p*p*p*p*p);
  deltaP = -dP;
  deltaCov00 = sigp2;
}
//
// Computation of energy loss for electrons
//
void 
VolumeEnergyLossEstimator::computeElectrons (const LocalVector& localP, double pathLength,
					     const VolumeMediumProperties& medium,
					     double& deltaP, double& deltaCov00) const {
  //
  // calculate absolute momentum and correction to path length from angle 
  // of incidence
  //
  double p = localP.mag();
  double normalisedPath = pathLength / medium.x0();
  //
  // Energy loss and variance according to Bethe and Heitler, see also
  // Comp. Phys. Comm. 79 (1994) 157. 
  //
  double z = exp(-normalisedPath);
  double varz = (exp(-normalisedPath*log(3.)/log(2.))-
  		 exp(-2*normalisedPath));
  // FIXME: need to know propagation direction at this point
//   if ( propDir==oppositeToMomentum ) {
//     //
//     // for backward propagation: delta(1/p) is linear in z=p_outside/p_inside
//     // convert to obtain equivalent delta(p). Sign of deltaP is corrected
//     // in method compute -> deltaP<0 at this place!!!
//     //
//     deltaP = -p*(1/z-1);
//     deltaCov00 = varz/p/p;
//   }
//   else {	
  //
  // for forward propagation: calculate in p (linear in 1/z=p_inside/p_outside),
  // then convert sig(p) to sig(1/p). 
  //
  deltaP = p*(z-1);
  //    double f = 1/p/z/z;
  // patch to ensure consistency between for- and backward propagation
  double f = 1./p/z;
  deltaCov00 += f*f*varz;
}
