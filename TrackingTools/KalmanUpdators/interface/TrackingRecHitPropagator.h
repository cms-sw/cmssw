#ifndef TrackingTools_TrackingRecHitPropagator_h
#define TrackingTools_TrackingRecHitPropagator_h

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


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

        /*
	template <class ResultingHit> TrackingRecHit::RecHitPointer project(const TrackingRecHit::ConstRecHitPointer hit,
                         					            const GeomDet& det,
                         					            const TrajectoryStateOnSurface ts) const{
	//1) propagate the best possible track parameters to the surface of the hit you want to "move" using a AnalyticalPropagator ;
	//2) create LocalTrajectoryParameters with the local x,y of the hit and direction + momentum from the propagated track parameters;
	//3) create a LocalTrajectoryError matrix which is 0 except for the local x,y submatrix, which is filled with the hit errors;
	//4) create a TSOS from the result of 2) and 3) and propagate it to the reference surface;
	//5) create a new hit with the local x,y subspace of the result of 4)
	  if (!ts.isValid()) return InvalidTransientRecHit::build(hit->det());
	  //	  LogTrace("SiTrackerMultiRecHitUpdator") << "the tsos is valid";	  
		//check if the ts lays or not on the destination surface and in case propagate it
		TrajectoryStateOnSurface propagated =ts;
		if (hit->surface() != &(ts.surface())) propagated = thePropagator->propagate(ts, *(hit->surface()));
		if (!propagated.isValid()) return InvalidTransientRecHit::build(hit->det());	
		//	  LogTrace("SiTrackerMultiRecHitUpdator") << "the propagate tsos is valid";	  
		//		LogTrace("SiTrackerMultiRecHitUpdator") << "Original: position: "<<hit->parameters()<<" error: "<<hit->parametersError()<<std::endl;
		//clone the original hit with this state
		TrackingRecHit::RecHitPointer updatedOriginal = hit->clone(propagated);
		//		LogTrace("SiTrackerMultiRecHitUpdator") << "New: position: "<<updatedOriginal->parameters()<<" error: "<<updatedOriginal->parametersError()<<std::endl;
		
		//	  LogTrace("SiTrackerMultiRecHitUpdator") << "rechit cloned";	  
		LocalTrajectoryParameters ltp(updatedOriginal->localPosition(), propagated.localMomentum(), propagated.charge());
		AlgebraicSymMatrix55 ltem;
		ltem(3,3) = (updatedOriginal->parametersError())(1,1);
		ltem(4,4) = (updatedOriginal->parametersError())(2,2);
		ltem(3,4) = (updatedOriginal->parametersError())(1,2);
		//		LogTrace("SiTrackerMultiRecHitUpdator") <<"The cov matrix: "<<ltem<<std::endl;
		LocalTrajectoryError lte(ltem);
		//		LogTrace("SiTrackerMultiRecHitUpdator") <<"Original cov matrix: "<<lte.matrix()<<std::endl;
		TrajectoryStateOnSurface hit_state(ltp, lte, propagated.surface(), propagated.magneticField());
		TrajectoryStateOnSurface projected_hit_state = thePropagator->propagate(hit_state, det.surface());
		if (!projected_hit_state.isValid()) return InvalidTransientRecHit::build(hit->det());	
		LocalPoint p = projected_hit_state.localPosition();
		LocalError e = projected_hit_state.localError().positionError();
		//		LogTrace("SiTrackerMultiRecHitUpdator") << "position: "<<p<<" error: "<<e<<std::endl;
		//AlgebraicSymMatrix55 projm=projected_hit_state.localError().matrix();	  
		//		for(int i=0;i<5;i++){
		//		LogTrace("SiTrackerMultiRecHitUpdator") <<"cov matrix: "<<projm<<std::endl;
		//		}
		return ResultingHit::build(p, e, &det, updatedOriginal->det(), updatedOriginal, this);
	}
	
        */
	private:
	const AnalyticalPropagator* thePropagator;
};

#endif
