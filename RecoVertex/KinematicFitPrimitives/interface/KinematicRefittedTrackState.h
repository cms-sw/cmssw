#ifndef KinematicRefittedTrackState_H
#define KinematicRefittedTrackState_H

#include "RecoVertex/VertexPrimitives/interface/RefittedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/PerigeeKinematicState.h"


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

class KinematicRefittedTrackState : public RefittedTrackState{

public:

 KinematicRefittedTrackState(const KinematicState& st);

/**
 * Access to Kinematic parameters
 */
 AlgebraicVector parameters() const;
  
/**
 * Kinmatic covariance
 */  
 AlgebraicSymMatrix covariance() const ;

/**
 * FTS out of kinematic parameters
 */
 FreeTrajectoryState freeTrajectoryState() const;
 
 GlobalPoint position() const;

/**
 * Kinematic momentum vector
 */
 AlgebraicVector kinematicMomentumVector() const;

/**
 * Perigee momentum vector
 */
 AlgebraicVector momentumVector() const;

 TrajectoryStateOnSurface trajectoryStateOnSurface(const Surface & surface) const;

 TrajectoryStateOnSurface trajectoryStateOnSurface(const Surface & surface, 
                                                   const Propagator & propagator) const;
						   
 virtual double weight() const;

 virtual ReferenceCountingPointer<RefittedTrackState> stateWithNewWeight
  	(const double newWeight) const;

 virtual std::vector< ReferenceCountingPointer<RefittedTrackState> > components() const;						   


private:

 KinematicState state; 

};
#endif
