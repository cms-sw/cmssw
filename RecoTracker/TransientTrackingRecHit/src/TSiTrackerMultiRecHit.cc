#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"

/*
TSiTrackerMultiRecHit::TSiTrackerMultiRecHit(const GeomDet * geom, const std::vector<const TrackingRecHit*>& rhs, const SiTrackerMultiRecHitUpdator* updator, const TrajectoryStateOnSurface& tsos):
TValidTrackingRecHit(geom), theUpdator(updator){
        theHitData  = theUpdator->buildMultiRecHit(tsos, rhs, theComponents);
	setAnnealingFactor(theUpdator->getCurrentAnnealingValue()); 
}
*/

const GeomDetUnit* TSiTrackerMultiRecHit::detUnit() const{
  return dynamic_cast<const GeomDetUnit*>(det());
}

TransientTrackingRecHit::RecHitPointer TSiTrackerMultiRecHit::clone(const TrajectoryStateOnSurface& ts) const{
/*
	std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents = theComponents;
	SiTrackerMultiRecHit better = theUpdator->update(ts,&theHitData, updatedcomponents);
      	RecHitPointer result = TSiTrackerMultiRecHit::build( det(), &better, theUpdator, updatedcomponents );
      	return result;
*/
	return RecHitPointer(this->clone());
}

/*

std::vector<const TrackingRecHit*> TSiTrackerMultiRecHit::recHits() const {
	std::vector<const TrackingRecHit*> components;
	std::vector<TransientTrackingRecHit::RecHitPointer>::const_iterator iter;
	for (iter = theComponents.begin(); iter != theComponents.end(); iter++){
		components.push_back(iter->get());
	}  
	return components;
}

*/
