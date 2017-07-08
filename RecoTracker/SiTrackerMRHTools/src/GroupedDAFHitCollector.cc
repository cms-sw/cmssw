#include "RecoTracker/SiTrackerMRHTools/interface/GroupedDAFHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MeasurementByLayerGrouper.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

#include <vector>
#include <map>

using namespace std;

vector<TrajectoryMeasurement> GroupedDAFHitCollector::recHits(const Trajectory& traj, 
							      const MeasurementTrackerEvent *theMTE) const
{

  LayerMeasurements theLM (theMTE->measurementTracker(), *theMTE);

  //WARNING: At the moment the trajectories has the measurements with reversed sorting after the track smoothing
  const vector<TrajectoryMeasurement>& meas = traj.measurements();
  const Propagator* forwardPropagator = getPropagator();
  const Propagator* backwardPropagator = getReversePropagator();
  if (traj.direction() == alongMomentum){
    forwardPropagator = getReversePropagator();
    backwardPropagator = getPropagator(); 
  }  
  if (meas.empty()) //return TransientTrackingRecHit::ConstRecHitContainer();	
    return vector<TrajectoryMeasurement>();

	//groups the TrajectoryMeasurements on a layer by layer
	vector<pair<const DetLayer*, vector<TrajectoryMeasurement> > > mol;
	mol = MeasurementByLayerGrouper(getMeasurementTracker()->geometricSearchTracker())(meas);

	vector<TrajectoryMeasurement> result;

	//add a protection if all the measurement are on the same layer
	if(mol.size()<2)return vector<TrajectoryMeasurement>();

	//it assumes that the measurements are sorted in the smoothing direction
	//TrajectoryStateOnSurface current = (*(mol.begin()+1)).second.front().updatedState();
	TrajectoryStateOnSurface current = (*(mol.rbegin()+1)).second.back().updatedState();
	//if (current.isValid()) current.rescaleError(10);

	
	//protection for layers with invalid meas with no id associated
	//to be fixed
	//for the moment no hit are lookerd for in these layers
	//remind that:
	//groupedMeasurements will return at least a measurement with an invalid hit with no detid
	LogDebug("MultiRecHitCollector") << "Layer "  << mol.back().first  << " has " << mol.back().second.size() << " measurements";
        LogTrace("MultiRecHitCollector") << "Original measurements are:";
        for( unsigned int iLay = 0; iLay < mol.size(); iLay++){
          LogTrace("MultiRecHitCollector") << "  Layer "  << mol.at(iLay).first  << " has " << mol.at(iLay).second.size() << " measurements:";
          vector<TrajectoryMeasurement>::const_iterator ibeg = (mol.at(iLay)).second.begin();
          vector<TrajectoryMeasurement>::const_iterator iend = (mol.at(iLay)).second.end();
          for (vector<TrajectoryMeasurement>::const_iterator imeas = ibeg; imeas != iend; ++imeas){
            if (imeas->recHit()->isValid()){
              LogTrace("MultiRecHitCollector") << "   Valid Hit with DetId " << imeas->recHit()->geographicalId().rawId()
                        << " local position " << imeas->recHit()->hit()->localPosition()
                        << " global position " << imeas->recHit()->hit()->globalPosition() ;
            } else {
              LogTrace("MultiRecHitCollector") << "   Invalid Hit with DetId " << imeas->recHit()->geographicalId().rawId(); 
            }
          }
        }

	//ERICA: I have to understand how are set the TM now. REPLACE THIS PART!!
        vector<TrajectoryMeasurementGroup> groupedMeas;
	if (mol.back().first) 
	  groupedMeas = theLM.groupedMeasurements(*(mol.back().first), current, 
						  *backwardPropagator, *(getEstimator()));

	//Since we have passed the backwardPropagator, we have to sort the detGroups in the opposite way 
	//(according the forward propagator, not the backward one)
	vector<TrajectoryMeasurementGroup> sortedgroupedMeas; 
	for (vector<TrajectoryMeasurementGroup>::reverse_iterator iter = groupedMeas.rbegin();  
	     iter != groupedMeas.rend(); iter++){

	  sortedgroupedMeas.push_back(*iter);

 	}

	//for the first layer
	buildMultiRecHits(sortedgroupedMeas, result, theMTE);
		

	//for other layers
	current = mol.back().second.front().updatedState();
	//if (current.isValid()) current.rescaleError(10);

	for( vector<pair<const DetLayer*, vector<TrajectoryMeasurement> > >::reverse_iterator imol = 
	      mol.rbegin() + 1; imol != mol.rend(); imol++) {
	
          const DetLayer* lay = (*imol).first;
	  LogDebug("MultiRecHitCollector") << "Layer "  << lay << " has " << (*imol).second.size() << " measurements";
          //debug 
	  vector<TrajectoryMeasurementGroup> currentLayerMeas;
	  if (lay) { 
	    currentLayerMeas = theLM.groupedMeasurements(*lay, current, *forwardPropagator, *(getEstimator()));
	  }

	  buildMultiRecHits(currentLayerMeas, result, theMTE);
	  current = (*imol).second.front().updatedState();
	  //if (current.isValid()) current.rescaleError(10);
	}

	LogTrace("MultiRecHitCollector") << " Ending GroupedDAFHitCollector::recHits >> Original Measurement size "  << meas.size() 
		  << "\n                                      >> GroupedDAFHitCollector returned " << result.size() << " measurements";
	//results are sorted in the fitting direction

 //	adding a protection against too few hits and invalid hits (due to failed propagation on the same surface of the original hits)
        if (result.size()>2)
          {
            int hitcounter=0;
            //check if the vector result has more than 3 valid hits
            for (vector<TrajectoryMeasurement>::const_iterator iimeas = result.begin(); iimeas != result.end(); ++iimeas)
            {
                if(iimeas->recHit()->isValid()) hitcounter++;
            }
            
            if(hitcounter>2) 
            {
		return result;
	    } 
            
            else return vector<TrajectoryMeasurement>();
          }
        
        else{return vector<TrajectoryMeasurement>();}
	
}

void GroupedDAFHitCollector::buildMultiRecHits(const vector<TrajectoryMeasurementGroup>& measgroup, vector<TrajectoryMeasurement>& result, const MeasurementTrackerEvent*& theMTE) const {

  unsigned int initial_size = result.size();

  //TransientTrackingRecHit::ConstRecHitContainer rhits;
  if (measgroup.empty()) {
    LogTrace("MultiRecHitCollector") << "No TrajectoryMeasurementGroups found for this layer\n" ; 
    //should we do something?
    //result.push_back(InvalidTransientRecHit::build(0,TrackingRecHit::missing));
    return;
  } 

  //we build a MultiRecHit out of each group
  //groups are sorted along momentum or opposite to momentum, 
  //measurements in groups are sorted with increating chi2
  LogTrace("MultiRecHitCollector") << "Found " << measgroup.size() << " groups for this layer";

  //trajectory state to store the last valid TrajectoryState (if present) to be used 
  //to add an invalid Measurement in case no valid state or no valid hits are found in any group
  for ( vector<TrajectoryMeasurementGroup>::const_iterator igroup = measgroup.begin(); 
	igroup != measgroup.end(); igroup++ ){

    //the TrajectoryState is the first one
    TrajectoryStateOnSurface state = igroup->measurements().front().predictedState();
    if (!state.isValid()){
      LogTrace("MultiRecHitCollector") << "Something wrong! no valid TSOS found in current group ";
      continue;		
    }

    LogTrace("MultiRecHitCollector") << "This group has " << igroup->measurements().size() << " measurements";
    LogTrace("MultiRecHitCollector") << "This group has the following " << igroup->detGroup().size() 
				     << " detector ids: " << endl;
    for (DetGroup::const_iterator idet = igroup->detGroup().begin(); idet != igroup->detGroup().end(); ++idet){
      LogTrace("MultiRecHitCollector") << idet->det()->geographicalId().rawId();
    }

    vector<const TrackingRecHit*> hits;
    for (vector<TrajectoryMeasurement>::const_iterator imeas = igroup->measurements().begin(); 
	 imeas != igroup->measurements().end(); imeas++){

      //should be fixed!!	
      //DetId id = imeas->recHit()->geographicalId();
      //MeasurementDetWithData measDet = theMTE->idToDet(id);

      //collect the non missing hits to build the MultiRecHits
      //we use the recHits method; anyway only simple hits, not MultiHits should be present 
      if (imeas->recHit()->getType() != TrackingRecHit::missing) {
        LogTrace("MultiRecHitCollector") << "This hit is valid ";
        hits.push_back(imeas->recHit()->hit());
      }
      else{
        LogTrace("MultiRecHitCollector") << "     This hit is not valid and will not enter in the MRH. " ;
      }
    }

    if (hits.empty()){
      LogTrace("MultiRecHitCollector") << "No valid hits found in current group ";
      continue;
    }

    LogTrace("MultiRecHitCollector") << "The best TSOS in this group is " << state << " it lays on surface located at " << state.surface().position();

    LogTrace("MultiRecHitCollector") << "For the MRH on this group the following hits will be used"; 
    for (vector<const TrackingRecHit*>::iterator iter = hits.begin(); iter != hits.end(); iter++){  
      string validity = "valid";
      if ((*iter)->getType() == TrackingRecHit::missing ) validity = "missing !should not happen!";
      else if ((*iter)->getType() == TrackingRecHit::inactive) validity = "inactive";
      else if ((*iter)->getType() == TrackingRecHit::bad) validity = "bad";   
      LogTrace("MultiRecHitCollector") << "DetId " << (*iter)->geographicalId().rawId()  
				       << " validity: " << validity 
				       << " surface position " << getMeasurementTracker()->geomTracker()->idToDet((*iter)->geographicalId())->position()  
				       << " hit local position " << (*iter)->localPosition();
    }
    //should be fixed!!
    //result.push_back(TrajectoryMeasurement(state,theUpdator->buildMultiRecHit(hits, state, *MeasurementDetWithData())));
  }
	//can this happen? it means that the measgroup was not empty but no valid measurement was found inside
	//in this case we add an invalid measuremnt for this layer 
	if (result.size() == initial_size){
		LogTrace("MultiRecHitCollector") << "no valid measuremnt or no valid TSOS in none of the groups";
		//measgroup has been already checked for size != 0
		if (!measgroup.back().measurements().empty()){	
			result.push_back(measgroup.back().measurements().back());
		} 
	}
}

