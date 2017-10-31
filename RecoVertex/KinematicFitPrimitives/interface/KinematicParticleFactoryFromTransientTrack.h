#ifndef KinematicParticleFactoryFromTransientTrack_h
#define KinematicParticleFactoryFromTransientTrack_h

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TrackKinematicStatePropagator.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicStateBuilder.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"

/**
 *.Factory for KinematicParticle RefCountedPointers
 */


class KinematicParticleFactoryFromTransientTrack
{
 public:

/**
 * Default constructoru sing a TrackKinematicStatePropagator  
 */ 
  KinematicParticleFactoryFromTransientTrack();
 
/**
 * Factory constructor taking a KinematicStatePropagator
 */ 
  KinematicParticleFactoryFromTransientTrack(KinematicStatePropagator * pr);
                                                      
/**
 * Default destructor
 */ 
  ~KinematicParticleFactoryFromTransientTrack()
  {delete propagator;}

/**
 * Particle constructed out of corresponding TransientTrack,
 * mass guess and sigma, chi2 and ndf. KinematicState 
 * is created at the point where TransientTrack is defined.
 */
  RefCountedKinematicParticle particle(const reco::TransientTrack& initialTrack, 
                                       const ParticleMass& massGuess,
				       float chiSquared, 
				       float degreesOfFr, 
                                       float& m_sigma) const;
                                       
/**
 * Particle constructed out of corresponding TransientTrack,
 * mass guess and sigma, chi2 and ndf. KinematicState 
 * is created from the given FreeTrajectoryState
 */
  RefCountedKinematicParticle particle(const reco::TransientTrack& initialTrack, 
                                       const ParticleMass& massGuess,
                                       float chiSquared, 
                                       float degreesOfFr, 
                                       float& m_sigma,
                                       const FreeTrajectoryState &freestate) const;                                       

/**
 * Particle is constructed out of corresponding TransientTrack,
 * mass_guess and sigma, chi2 and ndf. KinematicState is 
 * then propagated to the given point
 */
 
  RefCountedKinematicParticle particle(const reco::TransientTrack& initialTrack, 
                                       const ParticleMass& massGuess,
				       float chiSquared, 
				       float degreesOfFr,
				       const GlobalPoint& expPoint, 
			               float m_sigma) const;
 
/**
 * Particle is consructed directly from its KinematicState,
 * chi2 and related information. If no previous provided,
 * initial and currnt kinemtic states of the particle will match,
 * othereise, initial state will be taken from previous particle.
 */ 
  RefCountedKinematicParticle particle(const KinematicState& kineState, float& chiSquared,
                 float& ndf, ReferenceCountingPointer<KinematicParticle> previousParticle,
				         KinematicConstraint * lastConstraint = nullptr) const;

private:
  
 KinematicStatePropagator * propagator;
 TransientTrackKinematicStateBuilder builder;

};


#endif
