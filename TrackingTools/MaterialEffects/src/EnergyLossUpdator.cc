#include "TrackingTools/MaterialEffects/interface/EnergyLossUpdator.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"

#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"


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

  typedef float Float;

  Float p2 = localP.mag2();
  Float xf = std::abs(std::sqrt(p2)/localP.z());
   
  // constants
  const Float m2 = mass()*mass();           // use mass hypothesis from constructor

  constexpr Float emass = 0.511e-3;
  constexpr Float poti = 16.e-9 * 10.75; // = 16 eV * Z**0.9, for Si Z=14
  constexpr Float eplasma = 28.816e-9 * sqrt(2.33*0.498); // 28.816 eV * sqrt(rho*(Z/A)) for Si
  constexpr Float delta0 = 2*log(eplasma/poti) - 1.;

  // calculate general physics things
  Float im2 = Float(1.)/m2;
  Float e2     = p2 + m2;
  Float e = std::sqrt(e2);
  Float beta2  = p2/e2;
  Float eta2  = p2*im2;
  Float ratio2 = (emass*emass)*im2;
  Float emax  = Float(2.)*emass*eta2/(Float(1.) + Float(2.)*emass*e*im2 + ratio2);
  
  Float xi = materialConstants.xi()*xf; xi /= beta2;
  
  Float dEdx = xi*(unsafe_logf<2>(Float(2.)*emass*emax/(poti*poti)) - Float(2.)*(beta2) - delta0);
  
  Float dEdx2 = xi*emax*(Float(1.)-Float(0.5)*beta2);
  Float dP    = dEdx/std::sqrt(beta2);
  Float sigp2 = dEdx2/(beta2*p2*p2);
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
  float p2 = localP.mag2();
  float p = std::sqrt(p2);
  float normalisedPath = std::abs(p/localP.z())*materialConstants.radLen();
  //
  // Energy loss and variance according to Bethe and Heitler, see also
  // Comp. Phys. Comm. 79 (1994) 157. 
  //
  constexpr float l3ol2 = std::log(3.)/std::log(2.);
  float z = unsafe_expf<3>(-normalisedPath);
  float varz = unsafe_expf<3>(-normalisedPath*l3ol2)- 
                z*z;
  		// exp(-2*normalisedPath);

  if ( propDir==oppositeToMomentum ) {
    //
    // for backward propagation: delta(1/p) is linear in z=p_outside/p_inside
    // convert to obtain equivalent delta(p). Sign of deltaP is corrected
    // in method compute -> deltaP<0 at this place!!!
    //
    theDeltaP += -p*(1.f/z-1.f);
    theDeltaCov(0,0) += varz/p2;
  }
  else {	
    //
    // for forward propagation: calculate in p (linear in 1/z=p_inside/p_outside),
    // then convert sig(p) to sig(1/p). 
    //
    theDeltaP += p*(z-1.f);
    //    float f = 1/p/z/z;
    // patch to ensure consistency between for- and backward propagation
    float f2 = 1.f/(p2*z*z);
    theDeltaCov(0,0) += f2*varz;
  }
}
