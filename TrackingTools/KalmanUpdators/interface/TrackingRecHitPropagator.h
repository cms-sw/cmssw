#ifndef TrackingTools_TrackingRecHitPropagator_h
#define TrackingTools_TrackingRecHitPropagator_h

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

/* propagates the RecHit position from the original reference frame
   to the reference frame of another detector.
   Useful for algorithms like the DAF or the MTF	
*/

class TrackingRecHitPropagator {
	public: 
	TrackingRecHitPropagator(const MagneticField* magf){
		thePropagator = new AnalyticalPropagator(magf, anyDirection, 1.6);
	};

	~TrackingRecHitPropagator() {delete thePropagator;}

	template <class ResultingHit> TransientTrackingRecHit::RecHitPointer project(const TransientTrackingRecHit::ConstRecHitPointer hit,
                         							     const GeomDet& det,
                         					                     const TrajectoryStateOnSurface ts) const{
	//1) propagate the best possible track parameters to the surface of the hit you want to "move" using a AnalyticalPropagator ;
	//2) create LocalTrajectoryParameters with the local x,y of the hit and direction + momentum from the propagated track parameters;
	//3) create a LocalTrajectoryError matrix which is 0 except for the local x,y submatrix, which is filled with the hit errors;
	//4) create a TSOS from the result of 2) and 3) and propagate it to the reference surface;
	//5) create a new hit with the local x,y subspace of the result of 4)
	
		//check if the ts lays or not on the destination surface and in case propagate it
		TrajectoryStateOnSurface propagated = ts;
		if (hit->surface() != &(ts.surface())) propagated = thePropagator->propagate(ts, *(hit->surface()));
		//clone the original hit with this state
		TransientTrackingRecHit::RecHitPointer updatedOriginal = hit->clone(propagated);
		LocalTrajectoryParameters ltp(updatedOriginal->localPosition(), propagated.localMomentum(), propagated.charge());
		AlgebraicSymMatrix55 ltem;
		ltem(3,3) = (updatedOriginal->parametersError())(0,0);
		ltem(4,4) = (updatedOriginal->parametersError())(1,1);
		ltem(3,4) = (updatedOriginal->parametersError())(0,1);
		LocalTrajectoryError lte(ltem);
		TrajectoryStateOnSurface hit_state(ltp, lte, propagated.surface(), propagated.magneticField());
		TrajectoryStateOnSurface projected_hit_state = thePropagator->propagate(hit_state, det.surface());
		LocalPoint p = projected_hit_state.localPosition();
		LocalError e = projected_hit_state.localError().positionError();
		return ResultingHit::build(p, e, &det, updatedOriginal->det(), updatedOriginal, this);
	}
	
	private:
	const AnalyticalPropagator* thePropagator;
};

#endif
