#ifndef TransientTrackKinematicStateBuilder_H
#define TransientTrackKinematicStateBuilder_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TrackKinematicStatePropagator.h"
#include "RecoVertex/KinematicFitPrimitives/interface/PerigeeKinematicState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "MagneticField/Engine/interface/MagneticField.h"

/**
 * Kinematic State builder for TransientTrack  based kinematic states.
 * Uses TrackKinematicStatePropagator for state propagation.
 */

class TransientTrackKinematicStateBuilder
{

public:
 TransientTrackKinematicStateBuilder(){}
 
 ~TransientTrackKinematicStateBuilder(){}
 
/**
 * Operator creating a KinematcState at RecObj definition point
 * with given mass guess and sigma
 */ 
  KinematicState operator()(const reco::TransientTrack& track, const ParticleMass& m, float m_sigma) const ; 
 
 
 
/**
 * Operator creating a KinematicState directly out of 
 * 7 state parameters and their covariance matrix
 */ 
  KinematicState operator()(const KinematicParameters& par,
	const KinematicParametersError& er, const TrackCharge& ch,
	const MagneticField* field) const;
 
/**
 * Operator creating a KinematicState out of a RecObj
 * and propagating it to the given point using propagator
 * provided by user
 */ 
 KinematicState operator()(const reco::TransientTrack& track, const GlobalPoint& point, const ParticleMass& m,
                                                                             float m_sigma) const; 

/**
 * Operator to create a particle state at point
 * using the FreeTrajectoryState, charge and mass guess for the particle. The state will be
 * created with the reference point taken from the FTS
 */ 
 KinematicState operator()(const FreeTrajectoryState& state, const ParticleMass& mass,
                           float m_sigma) const;                                                                             
                                                                             
/**
 * Operator to create a particle state at point
 * using the FreeTrajectoryState, charge and mass guess for the particle. The state will be
 * created by propagating FTS to the transvese point of closest approach to the given point
 */ 
 KinematicState operator()(const FreeTrajectoryState& state, const ParticleMass& mass,
			   float m_sigma, const GlobalPoint& point) const;
			   
 PerigeeKinematicState operator()(const KinematicState& state, const GlobalPoint& point)const;											     
									     
private:

 KinematicState buildState(const FreeTrajectoryState & state, const ParticleMass& mass, 
                           float m_sigma)const;
 
 
 
 
 TrackKinematicStatePropagator propagator;
 
};
#endif
