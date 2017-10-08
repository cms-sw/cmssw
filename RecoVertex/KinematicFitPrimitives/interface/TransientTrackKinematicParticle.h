#ifndef TransientTrackKinematicParticle_H
#define TransientTrackKinematicParticle_H

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicStatePropagator.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleKinematicLinearizedTrackStateFactory.h"

/**
 * Class representing KinematicParticle
 * created fromTransientTrack. Uses specific
 * KinematicState builders and propagators.
 * Extends KinematicParticle class 
 * implementing its abstract methods and
 * adding several new ones.
 */
class TransientTrackKinematicParticle : public KinematicParticle
{
public:
 
/**
 * Constructor using KinematicState, Previous state of particle
 * last constraint used and original TransientTrack, if any.
 * All the pointers can be set to 0 if there's no such information
 * available. Constructor should be use by specific factory only.
 * Propagator for TransientTrack KinematicState is used.
 */  
 TransientTrackKinematicParticle(const KinematicState& kineState,float& chiSquared,
                       float& degreesOfFr,KinematicConstraint * lastConstraint,
                   ReferenceCountingPointer<KinematicParticle> previousParticle,
		   KinematicStatePropagator * pr,const reco::TransientTrack * initialTrack = nullptr);
		   	
 ~TransientTrackKinematicParticle() override;
 					   
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
 * Access to initial TransientTrack (if any)
 */  
 const reco::TransientTrack * initialTransientTrack() const;

/**
 * Method producing new particle with refitted parameters.
 * Current state is then shifted to previous state.
 * RCP<TransientTrackKinematicParticle> is  returned.
 */ 
 ReferenceCountingPointer<KinematicParticle> refittedParticle(const KinematicState& state,
                            float chi2, float ndf, KinematicConstraint * cons = nullptr)const override;
			    
/**
 * Method returning LinearizedTrackState of the particle needed for
 * Kalman flter vertex fit. This implementation uses the ParticleLinearizedTrackStateFactory class.
 */					    
 RefCountedLinearizedTrackState particleLinearizedTrackState(const GlobalPoint& point)const override; 


private: 

//initial TransientTrack (if any) 
 const reco::TransientTrack * inTrack;
  
//propagator for kinematic states
 KinematicStatePropagator * propagator;  

//LinearizedTrackStateFactory  specific for this
//type of particle
 ParticleKinematicLinearizedTrackStateFactory linFactory;
};
#endif
