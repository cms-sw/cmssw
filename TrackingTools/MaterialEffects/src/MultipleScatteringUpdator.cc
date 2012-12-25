#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"

#include "DataFormats/Math/interface/approx_log.h"

void oldMUcompute (const TrajectoryStateOnSurface& TSoS, 
		   const PropagationDirection propDir, double mass, double ptmin);

//#define DBG_MSU

//
// Computation of contribution of multiple scatterning to covariance matrix 
//   of local parameters based on Highland formula for sigma(alpha) in plane.
//
void MultipleScatteringUpdator::compute (const TrajectoryStateOnSurface& TSoS, 
					 const PropagationDirection propDir, Effect & effect) const
{
  //
  // Get surface
  //
  const Surface& surface = TSoS.surface();
  //
  //
  // Now get information on medium
  //
  const MediumProperties& mp = surface.mediumProperties();
  if unlikely(mp.radLen()==0) return;

  // Momentum vector
  LocalVector d = TSoS.localMomentum();
  float p2 = d.mag2();
  d *= 1.f/sqrt(p2);
  float xf = 1.f/std::abs(d.z());         // increase of path due to angle of incidence
  // calculate general physics things
  constexpr float amscon = 1.8496e-4;    // (13.6MeV)**2
  const float m2 = mass()*mass();            // use mass hypothesis from constructor
  float e2     = p2 + m2;
  float beta2  = p2/e2;
  // calculate the multiple scattering angle
  float radLen = mp.radLen()*xf;     // effective rad. length
  float sigt2 = 0.;                  // sigma(alpha)**2

  // Calculated rms scattering angle squared.
  float fact = 1.f + 0.038f*unsafe_logf<2>(radLen); fact *=fact;
  float a = fact/(beta2*p2);
  sigt2 = amscon*radLen*a;
  
  if (thePtMin > 0) {
#ifdef DBG_MSU
    std::cout<<"Original rms scattering = "<<sqrt(sigt2);
#endif
    // Inflate estimated rms scattering angle, to take into account 
    // that 1/p is not known precisely.
    AlgebraicSymMatrix55 const & covMatrix = TSoS.localError().matrix();
    float error2_QoverP = covMatrix(0,0);
    // Formula valid for ultra-relativistic particles.
    //      sigt2 *= (1. + p2 * error2_QoverP);
    // Exact formula
    sigt2 *= 1.f +  error2_QoverP *( p2 + m2*beta2*(5.f + 3.f*p2*error2_QoverP) ) ;
#ifdef DBG_MSU
    std::cout<<" new = "<<sqrt(sigt2);
#endif
    // Convert Pt constraint to P constraint, neglecting uncertainty in 
    // track angle.
    float pMin2 = thePtMin*thePtMin*(p2/TSoS.globalMomentum().perp2());       
    // Use P constraint to calculate rms maximum scattering angle.
    float betaMin2 = pMin2/(pMin2 + m2);
    float a_max = fact/(betaMin2 * pMin2);
    float sigt2_max = amscon*radLen*a_max;
    if (sigt2 > sigt2_max) sigt2 = sigt2_max;
#ifdef DBG_MSU
    std::cout<<" after P constraint ("<<pMin<<") = "<<sqrt(sigt2);
    std::cout<<" for track with 1/p="<<1/p<<"+-"<<sqrt(error2_QoverP)<<std::endl;
#endif
  }
  
  float isl2 = 1.f/d.perp2();
  float cl2 = (d.z()*d.z());
  float cf2 = (d.x()*d.x())*isl2;
  float sf2 = (d.y()*d.y())*isl2;
  // Create update (transformation of independant variations
  //   on angle in orthogonal planes to local parameters.
  float den = 1.f/(cl2*cl2);
  using namespace materialEffect;
  effect.deltaCov[msxx] += (den*sigt2)*(sf2*cl2 + cf2);
  effect.deltaCov[msxy] += (den*sigt2)*(d.x()*d.y()  );
  effect.deltaCov[msyy] += (den*sigt2)*(cf2*cl2 + sf2);
  
  /*
    std::cout << "new " <<  theDeltaCov(1,1) << " " <<  theDeltaCov(1,2)  << " " <<  theDeltaCov(2,2) << std::endl;
    oldMUcompute(TSoS,propDir, mass(), thePtMin);
  */
    
}
    
    
    

//
// Computation of contribution of multiple scatterning to covariance matrix 
//   of local parameters based on Highland formula for sigma(alpha) in plane.
//
void oldMUcompute (const TrajectoryStateOnSurface& TSoS, 
		   const PropagationDirection propDir, float mass, float thePtMin)
{
  //
  // Get surface
  //
  const Surface& surface = TSoS.surface();
  //
  //
  // Now get information on medium
  //
  if (surface.mediumProperties()) {
    // Momentum vector
    LocalVector d = TSoS.localMomentum();
    float p = d.mag();
    d *= 1./p;
    // MediumProperties mp(0.02, .5e-4);
    const MediumProperties& mp = *surface.mediumProperties();
    float xf = 1./fabs(d.z());         // increase of path due to angle of incidence
    // calculate general physics things
    const float amscon = 1.8496e-4;    // (13.6MeV)**2
    const float m = mass;            // use mass hypothesis from constructor
    float e     = sqrt(p*p + m*m);
    float beta  = p/e;
    // calculate the multiple scattering angle
    float radLen = mp.radLen()*xf;     // effective rad. length
    float sigt2 = 0.;                  // sigma(alpha)**2
    if (radLen > 0) {
      // Calculated rms scattering angle squared.
      float a = (1. + 0.038*log(radLen))/(beta*p); a *= a;
      sigt2 = amscon*radLen*a;
      if (thePtMin > 0) {
#ifdef DBG_MSU
        std::cout<<"Original rms scattering = "<<sqrt(sigt2);
#endif
        // Inflate estimated rms scattering angle, to take into account 
        // that 1/p is not known precisely.
        AlgebraicSymMatrix55 covMatrix = TSoS.localError().matrix();
        float error2_QoverP = covMatrix(0,0);
	// Formula valid for ultra-relativistic particles.
//      sigt2 *= (1. + (p*p) * error2_QoverP);
	// Exact formula
        sigt2 *= (1. + (p*p) * error2_QoverP *
		               (1. + 5*m*m/(e*e) + 3*m*m*beta*beta*error2_QoverP));
#ifdef DBG_MSU
	std::cout<<" new = "<<sqrt(sigt2);
#endif
	// Convert Pt constraint to P constraint, neglecting uncertainty in 
	// track angle.
	float pMin = thePtMin*(TSoS.globalMomentum().mag()/TSoS.globalMomentum().perp());       
        // Use P constraint to calculate rms maximum scattering angle.
        float betaMin = pMin/sqrt(pMin * pMin + m*m);
        float a_max = (1. + 0.038*log(radLen))/(betaMin * pMin); a_max *= a_max;
        float sigt2_max = amscon*radLen*a_max;
        if (sigt2 > sigt2_max) sigt2 = sigt2_max;
#ifdef DBG_MSU
	std::cout<<" after P constraint ("<<pMin<<") = "<<sqrt(sigt2);
	std::cout<<" for track with 1/p="<<1/p<<"+-"<<sqrt(error2_QoverP)<<std::endl;
#endif
      }
    }
    float sl = d.perp();
    float cl = d.z();
    float cf = d.x()/sl;
    float sf = d.y()/sl;
    // Create update (transformation of independant variations
    //   on angle in orthogonal planes to local parameters.
    std::cout << "old " << sigt2*(sf*sf*cl*cl + cf*cf)/(cl*cl*cl*cl)
	      << " " << sigt2*(cf*sf*sl*sl        )/(cl*cl*cl*cl) 
	      << " " << sigt2*(cf*cf*cl*cl + sf*sf)/(cl*cl*cl*cl) << std::endl;
  }
}
