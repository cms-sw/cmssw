#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MeasurementDet/interface/GeometricSearchDetMeasurements.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "TrackingTools/DetLayers/interface/DetGroup.h"

using namespace std;

vector<TrajectoryMeasurement>
LayerMeasurements::measurements( const DetLayer& layer, 
				 const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop, 
				 const MeasurementEstimator& est) const
{
  typedef DetLayer::DetWithState   DetWithState;
  vector<DetWithState> compatDets = layer.compatibleDets( startingState, prop, est);

  vector<TrajectoryMeasurement> result;
  if (compatDets.empty()) {
    pair<bool, TrajectoryStateOnSurface> compat =
      layer.compatible( startingState, prop, est);

    if ( compat.first) {
      result.push_back( TrajectoryMeasurement( compat.second, 
					       InvalidTransientRecHit::build(0), 0.F,
					       &layer));
    }
    return result;
  }

  GeometricSearchDetMeasurements gsdm( theDetSystem);
  vector<TrajectoryMeasurement> tmpResult = gsdm.get( layer, compatDets, startingState, prop, est);

  for(vector<TrajectoryMeasurement>::const_iterator tmpIt=tmpResult.begin();tmpIt!=tmpResult.end();tmpIt++){
    result.push_back(  TrajectoryMeasurement(tmpIt->predictedState(),tmpIt->recHit(),tmpIt->estimate(),&layer)  );
  }
  
  return result;
}


vector<TrajectoryMeasurementGroup>
LayerMeasurements::groupedMeasurements( const DetLayer& layer, 
					const TrajectoryStateOnSurface& startingState,
					const Propagator& prop, 
					const MeasurementEstimator& est) const
{
  vector<TrajectoryMeasurementGroup> result;

  vector<DetGroup> groups( layer.groupedCompatibleDets( startingState, prop, est));
  for (vector<DetGroup>::const_iterator grp=groups.begin(); grp!=groups.end(); grp++) {
    if ( grp->empty() )  continue;

    vector<TrajectoryMeasurement> tmpVec;
    for (DetGroup::const_iterator idet=grp->begin(); idet!=grp->end(); idet++) {
      const MeasurementDet* mdet = theDetSystem->idToDet(idet->det()->geographicalId());
      if (mdet == 0) {
	throw MeasurementDetException( "MeasurementDet not found");
      }      
      vector<TrajectoryMeasurement> tmp = 
	mdet->fastMeasurements( idet->trajectoryState(), startingState, prop, est);
      if (!tmp.empty()) {
	// only collect valid RecHits
	vector<TrajectoryMeasurement>::iterator end = 
	  (tmp.back().recHit()->isValid() ? tmp.end() : tmp.end()-1);
	tmpVec.insert( tmpVec.end(), tmp.begin(), end);
      }
    }
    // sort the final result
    if ( static_cast<int>(tmpVec.size()) > 1) {
      sort( tmpVec.begin(), tmpVec.end(), TrajMeasLessEstim());
    }
    addInvalidMeas( tmpVec, *grp,layer); 
    result.push_back( TrajectoryMeasurementGroup( tmpVec, *grp));
  }

  // if the result is empty check if the layer is compatible (for invalid measurement)
  if (result.empty()) {
    pair<bool, TrajectoryStateOnSurface> compat = layer.compatible( startingState, prop, est);
    if ( compat.first) {
      TrajectoryMeasurement inval( compat.second, InvalidTransientRecHit::build(0), 0.F,&layer);
      vector<TrajectoryMeasurement> tmVec(1,inval);
      result.push_back( TrajectoryMeasurementGroup( tmVec, DetGroup()));
    }
  }
  return result;
}

void LayerMeasurements::addInvalidMeas( vector<TrajectoryMeasurement>& measVec,
					const DetGroup& group,
					const DetLayer& layer) const
{
  if (!measVec.empty()) {
    // invalidMeas on Det of most compatible hit
    measVec.push_back( TrajectoryMeasurement( measVec.front().predictedState(), 
					      InvalidTransientRecHit::build(measVec.front().recHit()->det()),
					      0.,&layer));
  }
  else if (!group.empty()) {
    // invalid state on first compatible Det
    measVec.push_back( TrajectoryMeasurement( group.front().trajectoryState(), 
					      InvalidTransientRecHit::build(group.front().det()), 0.,&layer));
  }
}
