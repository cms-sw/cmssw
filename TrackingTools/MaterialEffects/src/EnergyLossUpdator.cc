#include "TrackingTools/MaterialEffects/interface/EnergyLossUpdator.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"

//
// Computation of contribution of energy loss to momentum and covariance
// matrix of local parameters based on Bethe-Bloch. For electrons 
// contribution of radiation acc. to Bethe & Heitler.
//
void EnergyLossUpdator::compute (const TrajectoryStateOnSurface& TSoS, 
				 const PropagationDirection propDir) const
{
  //
  // Get surface
  //
  const Surface& surface = TSoS.surface();
  //
  // Initialise dP and the update to the covariance matrix
  //
  theDeltaP = 0.;
  theDeltaCov(0,0) = 0.;
  //
  // Now get information on medium
  //
  if (surface.mediumProperties()) {
    //
    // Bethe-Bloch
    //
    if ( mass()>0.001 )
      computeBetheBloch(TSoS.localMomentum(),*surface.mediumProperties());
    //
    // Special treatment for electrons (currently rather crude
    // distinction using mass)
    //
    else
      computeElectrons(TSoS.localMomentum(),*surface.mediumProperties(),
		       propDir);
    if (propDir != alongMomentum) theDeltaP *= -1.;
  }
}
//
// Computation of energy loss according to Bethe-Bloch
//
void
EnergyLossUpdator::computeBetheBloch (const LocalVector& localP,
				      const MediumProperties& materialConstants) const {
  //
  // calculate absolute momentum and correction to path length from angle 
  // of incidence
  //
  double p = localP.mag();
  double xf = fabs(p/localP.z());
  // constants
  const double m = mass();            // use mass hypothesis from constructor

  const double emass = 0.511e-3;
  const double poti = 16.e-9 * 10.75; // = 16 eV * Z**0.9, for Si Z=14
  const double eplasma = 28.816e-9 * sqrt(2.33*0.498); // 28.816 eV * sqrt(rho*(Z/A)) for Si
  const double delta0 = 2*log(eplasma/poti) - 1.;

  // calculate general physics things
  double e     = sqrt(p*p + m*m);
  double beta  = p/e;
  double gamma = e/m;
  double eta2  = beta*gamma; eta2 *= eta2;
  //  double lnEta2 = log(eta2);
  double ratio = emass/m;
  double emax  = 2.*emass*eta2/(1. + 2.*ratio*gamma + ratio*ratio);
  // double delta = delta0 + lnEta2;

  // calculate the mean and sigma of energy loss
  // xi = d[g/cm2] * 0.307075MeV/(g/cm2) * Z/A * 1/2
  double xi = materialConstants.xi()*xf; xi /= (beta*beta);

//   double dEdx = xi*(log(2.*emass*eta2*emax/(poti*poti)) - 2.*(beta*beta));
  //double dEdx = xi*(log(2.*emass*emax/(poti*poti))+lnEta2 - 2.*(beta*beta) - delta);

  double dEdx = xi*(log(2.*emass*emax/(poti*poti)) - 2.*(beta*beta) - delta0);
  double dEdx2 = xi*emax*(1.-0.5*(beta*beta));
  double dP    = dEdx/beta;
  double sigp2 = dEdx2*e*e/(p*p*p*p*p*p);
  theDeltaP += -dP;
  theDeltaCov(0,0) += sigp2;
}
//
// Computation of energy loss for electrons
//
void 
EnergyLossUpdator::computeElectrons (const LocalVector& localP,
				     const MediumProperties& materialConstants,
				     const PropagationDirection propDir) const {
  //
  // calculate absolute momentum and correction to path length from angle 
  // of incidence
  //
  double p = localP.mag();
  double normalisedPath = fabs(p/localP.z())*materialConstants.radLen();
  //
  // Energy loss and variance according to Bethe and Heitler, see also
  // Comp. Phys. Comm. 79 (1994) 157. 
  //
  double z = exp(-normalisedPath);
  double varz = exp(-normalisedPath*log(3.)/log(2.))- 
                z*z;
  		// exp(-2*normalisedPath);

  if ( propDir==oppositeToMomentum ) {
    //
    // for backward propagation: delta(1/p) is linear in z=p_outside/p_inside
    // convert to obtain equivalent delta(p). Sign of deltaP is corrected
    // in method compute -> deltaP<0 at this place!!!
    //
    theDeltaP += -p*(1/z-1);
    theDeltaCov(0,0) += varz/(p*p);
  }
  else {	
    //
    // for forward propagation: calculate in p (linear in 1/z=p_inside/p_outside),
    // then convert sig(p) to sig(1/p). 
    //
    theDeltaP += p*(z-1);
    //    double f = 1/p/z/z;
    // patch to ensure consistency between for- and backward propagation
    double f = 1./(p*z);
    theDeltaCov(0,0) += f*f*varz;
  }
}
