#ifndef KinematicRefittedTrackState_H
#define KinematicRefittedTrackState_H

#include "RecoVertex/VertexPrimitives/interface/RefittedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/PerigeeKinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"


/**
 * Refitted state for kinematic parameters
 * To be used in KinematicParticleVertxFitter
 * only. Class caches the refitted state of the
 * particle and provide access to its
 * parameters in both parametrizations:
 * Kinemaic and extended Perigee.
 *
 * Several methods are done just to be
 * consistent with present KalmanVertexFitter
 * structure
 */

class KinematicRefittedTrackState : public RefittedTrackState<6>{

public:

 typedef ReferenceCountingPointer<RefittedTrackState<6> > RefCountedRefittedTrackState;

 KinematicRefittedTrackState(const KinematicState& st, const AlgebraicVector4& mv);

/**
 * Access to Kinematic perigee parameters
 */
 AlgebraicVector6 parameters() const;
  
/**
 * Kinmatic perigee covariance
 */  
 AlgebraicSymMatrix66 covariance() const ;

/**
 * Access to Kinematic parameters
 */
 AlgebraicVector7 kinematicParameters() const;
  
/**
 * Kinmatic covariance
 */  
 AlgebraicSymMatrix77 kinematicParametersCovariance() const ;

/**
 * FTS out of kinematic parameters
 */
 FreeTrajectoryState freeTrajectoryState() const;
 
 GlobalPoint position() const;

/**
 * Kinematic momentum vector
 */
 AlgebraicVector4 kinematicMomentumVector() const;

/**
 * Perigee momentum vector
 */
 AlgebraicVector4 momentumVector() const;

 TrajectoryStateOnSurface trajectoryStateOnSurface(const Surface & surface) const;

 TrajectoryStateOnSurface trajectoryStateOnSurface(const Surface & surface, 
                                                   const Propagator & propagator) const;
						   
 virtual double weight() const;

 virtual ReferenceCountingPointer<RefittedTrackState<6> > stateWithNewWeight
  	(const double newWeight) const;

 virtual std::vector< ReferenceCountingPointer<RefittedTrackState<6> > > components() const;						   

 virtual reco::TransientTrack transientTrack() const;


private:

 KinematicState state; 
 AlgebraicVector4 momentumAtVertex;

};
#endif
