#include "RecoTracker/SiTrackerMRHTools/interface/SimpleDAFHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"

#include <vector>
#include <map>

#define _debug_SimpleDAFHitCollector_ 

using namespace std;

vector<TrajectoryMeasurement> SimpleDAFHitCollector::recHits(const Trajectory& traj) const{

	//it assumes that the measurements are sorted in the smoothing direction
	vector<TrajectoryMeasurement> meas = traj.measurements();
	if (meas.empty()) //return TransientTrackingRecHit::ConstRecHitContainer();	
		return vector<TrajectoryMeasurement>();


	//TransientTrackingRecHit::ConstRecHitContainer result;
	vector<TrajectoryMeasurement> result;

	for(vector<TrajectoryMeasurement>::reverse_iterator imeas = meas.rbegin(); imeas != meas.rend(); imeas++) {
		//if the rechit is associated to a valid detId  
		if(imeas->recHit()->geographicalId().rawId()){
			vector<TrajectoryMeasurement> currentLayerMeas = getMeasurementTracker()->idToDet(imeas->recHit()->geographicalId().rawId())->fastMeasurements(imeas->updatedState(), imeas->updatedState(), *thePropagator, *(getEstimator()));
	        	buildMultiRecHits(currentLayerMeas, result);
		} else {
			result.push_back(*imeas);
		}
	}
	LogTrace("MultiRecHitCollector") << "Original Measurement size "  << meas.size() << " SimpleDAFHitCollector returned " << result.size() << " rechits";
	//results are sorted in the fitting direction
	return result;	
}

void SimpleDAFHitCollector::buildMultiRecHits(const vector<TrajectoryMeasurement>& vmeas, vector<TrajectoryMeasurement>& result) const {

	if (vmeas.empty()) {
		LogTrace("MultiRecHitCollector") << "fastMeasurements returned an empty vector, should not happen " ; 
		//should we do something?
		//result.push_back(InvalidTransientRecHit::build(0,TrackingRecHit::missing));
		return;
	} 

	TrajectoryStateOnSurface state = vmeas.front().predictedState();

	if (state.isValid()==false){
		LogTrace("MultiRecHitCollector") << "first state is invalid; skipping ";
		return;
	}

	vector<const TrackingRecHit*> hits;
	//	TransientTrackingRecHit::ConstRecHitContainer hits;

	for (vector<TrajectoryMeasurement>::const_iterator imeas = vmeas.begin(); imeas != vmeas.end(); imeas++){
		if (imeas->recHit()->getType() != TrackingRecHit::missing) {
                                LogTrace("MultiRecHitCollector") << "This hit is valid ";
                                hits.push_back(imeas->recHit()->hit());
                }
	}

	if (hits.empty()){
		LogTrace("MultiRecHitCollector") << "No valid hits found ";
		return;
	}

	result.push_back(TrajectoryMeasurement(state,theUpdator->buildMultiRecHit(hits, state)));	
	//	result.push_back(TrajectoryMeasurement(state,theUpdator->update(hits, state)));	

}


