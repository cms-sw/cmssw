#include "RecoTracker/SiTrackerMRHTools/interface/GroupedDAFHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MeasurementByLayerGrouper.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

#include <vector>
#include <map>

#define _debug_GroupedDAFHitCollector_ 

using namespace std;

vector<TrajectoryMeasurement> GroupedDAFHitCollector::recHits(const Trajectory& traj) const{

	//it assumes that the measurements are sorted in the smoothing direction
	vector<TrajectoryMeasurement> meas = traj.measurements();
	const Propagator* forwardPropagator = getPropagator();
	const Propagator* backwardPropagator = getReversePropagator();
	if (traj.direction() == alongMomentum){
		forwardPropagator = getReversePropagator();
		backwardPropagator = getPropagator(); 
	}  
	if (meas.empty()) //return TransientTrackingRecHit::ConstRecHitContainer();	
		return vector<TrajectoryMeasurement>();

	vector<pair<const DetLayer*, vector<TrajectoryMeasurement> > > mol = MeasurementByLayerGrouper(getMeasurementTracker()->geometricSearchTracker())(meas);

	//TransientTrackingRecHit::ConstRecHitContainer result;
	vector<TrajectoryMeasurement> result;

	//first layer
	//  cout<<"DAFHitCollectionFromRecTrack: first layer"<<endl;

	//it assumes that the measurements are sorted in the smoothing direction
	//TrajectoryStateOnSurface current = (*(mol.begin()+1)).second.front().updatedState();
	TrajectoryStateOnSurface current = (*(mol.rbegin()+1)).second.back().updatedState();

	
	vector<TrajectoryMeasurementGroup> groupedMeas;
	//protection for layers with invalid meas with no id associated
	//to be fixed
	//for the moment no hit are lookerd for in these layers
	//remind that:
	//groupedMeasurements will return at least a measurement with an invalid hit with no detid
	LogDebug("MultiRecHitCollector") << "Layer "  << mol.front().first  << " has " << mol.front().second.size() << " measurements"; 	 
	if (mol.front().first) groupedMeas = theLM.groupedMeasurements(*(mol.back().first), current, *backwardPropagator, *(getEstimator()));
	//in this case we have to sort the detGroups in the opposite way (according the forward propagator, not the backward one)
	vector<TrajectoryMeasurementGroup> sortedgroupedMeas; 
	for (vector<TrajectoryMeasurementGroup>::reverse_iterator iter = groupedMeas.rbegin();  iter != groupedMeas.rend(); iter++){
		sortedgroupedMeas.push_back(*iter);
	}
	//TransientTrackingRecHit::ConstRecHitContainer rhits = buildMultiRecHits(groupedMeas);
	buildMultiRecHits(sortedgroupedMeas, result);
		

	//result.insert(result.end(),rhits.begin(),rhits.end());

	//other layers
	current = mol.back().second.front().updatedState();
	//for(vector<pair<const DetLayer*, vector<TrajectoryMeasurement> > >::iterator imol = mol.begin() + 1; imol != mol.end(); imol++) {
	for(vector<pair<const DetLayer*, vector<TrajectoryMeasurement> > >::reverse_iterator imol = mol.rbegin() + 1; imol != mol.rend(); imol++) {
		const DetLayer* lay = (*imol).first;
		LogDebug("MultiRecHitCollector") << "Layer "  << lay << " has " << (*imol).second.size() << " measurements"; 	 
		vector<TrajectoryMeasurementGroup> currentLayerMeas;
		if (mol.front().first) currentLayerMeas = theLM.groupedMeasurements(*lay, current, *forwardPropagator, *(getEstimator()));
	        //TransientTrackingRecHit::ConstRecHitContainer curhits = buildMultiRecHits(currentLayerMeas);
	        buildMultiRecHits(currentLayerMeas, result);
		//result.insert(result.end(),rhits.begin(),rhits.end()); 
		current= (*imol).second.front().updatedState();
	}
	LogTrace("MultiRecHitCollector") << "Original Measurement size "  << meas.size() << " GroupedDAFHitCollector returned " << result.size() << " rechits";
	//results are sorted in the fitting direction
	return result;	
}

void GroupedDAFHitCollector::buildMultiRecHits(const vector<TrajectoryMeasurementGroup>& measgroup, vector<TrajectoryMeasurement>& result) const {

	unsigned int initial_size = result.size();
	//TransientTrackingRecHit::ConstRecHitContainer rhits;
	if (measgroup.empty()) {
		LogTrace("MultiRecHitCollector") << "This is a layer with only an invalid hit with no detid associated" ; 
		//should we do something?
		//result.push_back(InvalidTransientRecHit::build(0,TrackingRecHit::missing));
		return;
	} 

	//we build a MultiRecHit out of each group
	//groups are sorted along momentum of opposite to momentum, 
	//measurements in groups are sorted with increating chi2
	LogTrace("MultiRecHitCollector") << "Found " << measgroup.size() << " groups for this layer";
	//trajectory state to store the last valid TrajectoryState (if present) to be used 
	//to add an invalid Measurement in case no valid state or no valid hits are found in any group
	TrajectoryStateOnSurface cachedstate;
	uint32_t cacheddetid=0;
	for (vector<TrajectoryMeasurementGroup>::const_iterator igroup = measgroup.begin(); igroup != measgroup.end(); igroup++ ){
		//the TrajectoryState is the first one
		TrajectoryStateOnSurface state = igroup->measurements().front().predictedState();
		if (!state.isValid()){
			LogTrace("MultiRecHitCollector") << "Something wrong! no valid TSOS found in current group ";
                        //is the following the right thing?
                        //rhits.push_back(InvalidTransientRecHit::build(igroup->detGroup().front().det(), TrackingRecHit::missing));
                        //result.push_back(InvalidTransientRecHit::build(igroup->detGroup().front().det(), TrackingRecHit::missing));
                        continue;		
		}
		//every time a valid tsoso is found 
		//the cached state and detid are updated	
		cachedstate = state;
		cacheddetid = igroup->measurements().front().recHit()->geographicalId().rawId();
		vector<const TrackingRecHit*> hits;
				//	TransientTrackingRecHit::ConstRecHitContainer hits;
		LogTrace("MultiRecHitCollector") << "This group has " << igroup->measurements().size() << " measurements";
		for (vector<TrajectoryMeasurement>::const_iterator imeas = igroup->measurements().begin(); imeas != igroup->measurements().end(); imeas++){
			//collect the non missing hits to build the MultiRecHits
			//we ese the recHits method; anyway only simple hits, not MultiHits should be present 
			if (imeas->recHit()->getType() != TrackingRecHit::missing) {
				LogTrace("MultiRecHitCollector") << "This hit is valid ";
				hits.push_back(imeas->recHit()->hit());
			}
		}
		if (hits.empty()){
			LogTrace("MultiRecHitCollector") << "No valid hits found in current group ";
			//is the following the right thing?
			//result.push_back(TrajectoryMeasurement(state,InvalidTransientRecHit::build(igroup->detGroup().front().det()),0.F));
			continue;
		}
		LogTrace("MultiRecHitCollector") << "The best TSOS in this group is " << state << " it lays on surface located at " << state.surface().position();
#ifdef _debug_GroupedDAFHitCollector_	
		LogTrace("MultiRecHitCollector") << "For the MRH on this group the following hits will be used"; 
		for (vector<const TrackingRecHit*>::iterator iter = hits.begin(); iter != hits.end(); iter++){  
		  //		for (TransientTrackingRecHit::ConstRecHitContainer::iterator iter = hits.begin(); iter != hits.end(); iter++){  
			string validity = "valid";
			if ((*iter)->getType() == TrackingRecHit::missing ) validity = "missing !should not happen!";
			else if ((*iter)->getType() == TrackingRecHit::inactive) validity = "inactive";
			else if ((*iter)->getType() == TrackingRecHit::bad) validity = "bad";   
			LogTrace("MultiRecHitCollector") << "DetId " << (*iter)->geographicalId().rawId()  << " validity: " << validity 
				<< " surface position " << getMeasurementTracker()->geomTracker()->idToDet((*iter)->geographicalId())->position()  
				<< " hit local position " << (*iter)->localPosition();
		}
#endif
		
		//result.push_back(TrajectoryMeasurement(state,TSiTrackerMultiRecHit::build(getMeasurementTracker()->geomTracker()->idToDet(hits.front()->geographicalId()), hits, theUpdator, state)));
		result.push_back(TrajectoryMeasurement(state,theUpdator->buildMultiRecHit(hits, state)));
	}

	//can this happen? it means that the measgroup was not empty but no valid measurement was found inside
	//in this case we add an invalid measuremnt for this layer 
	if (result.size() == initial_size){
		LogTrace("MultiRecHitCollector") << "no valid measuremnt or no valid TSOS in none of the groups";
		//should we do something?
		result.push_back(TrajectoryMeasurement(cachedstate, 
						       InvalidTransientRecHit::build(cacheddetid > 0 ? 
										     getMeasurementTracker()->geomTracker()->idToDet(DetId(cacheddetid)): 
										     0), 0.F));
	}
	//return rhits;
}


