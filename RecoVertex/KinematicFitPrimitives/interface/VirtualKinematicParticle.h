#ifndef VirtualKinematicParticle_H
#define VirtualKinematicParticle_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicStatePropagator.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleKinematicLinearizedTrackStateFactory.h"

/**
 * Class representing KinematicParticle
 * created as a result of kinematic fit.
 * uses the same state propagator as a 
 * TransientTrackKinematicParticle
 */
class VirtualKinematicParticle:public KinematicParticle
{
 public:

/**
 * Constructor using KinematicState, Previous state of particle
 * and last constraint used .
 * All the pointers can be set to 0 if there's no such information
 * available. Constructor should be use by specific factory only.
 * Propagator for TransientTrackKinematicState is used.
 */  
  VirtualKinematicParticle(const KinematicState& kineState,float& chiSquared,
                         float& degreesOfFr, KinematicConstraint * lastConstraint,
                     ReferenceCountingPointer<KinematicParticle> previousParticle,
		                                   KinematicStatePropagator * pr);
 
 ~VirtualKinematicParticle() override;
 
/**
 * Comparison by contents operator
 * Returns TRUE if initial TransientTracks
 * match(if they exist). If not, 
 * compares the initial KinematicStates
 * Retunes true if they match.
 */ 
  bool operator==(const KinematicParticle& other)const override;

  bool operator==(const ReferenceCountingPointer<KinematicParticle>& other) const override;

  bool operator!=(const KinematicParticle& other)const override;

/**
 * Access to KinematicState of particle
 * at given point
 */ 
 KinematicState stateAtPoint(const GlobalPoint& point)const override;
 
/**
 * Method producing new particle with refitted parameters.
 * Current state is then shifted to previous state.
 * RCP<VirtualKinematicParticle> is  returned.
 */ 
 RefCountedKinematicParticle refittedParticle(const KinematicState& state,
                               float chi2, float ndf, KinematicConstraint * cons = nullptr)const override;
			       
/**
 * Method returning LinearizedTrackState of the particle needed for
 * Kalman flter vertex fit. This implementation uses the ParticleLinearizedTrackStateFactory class.
 */					    
 RefCountedLinearizedTrackState particleLinearizedTrackState(const GlobalPoint& point)const override; 
			        
 private:
 
//propagator for kinematic states
 KinematicStatePropagator * propagator;  
 
//LinearizedTrackStateFactory  specific for this
//type of particle
 ParticleKinematicLinearizedTrackStateFactory linFactory;
};
#endif
