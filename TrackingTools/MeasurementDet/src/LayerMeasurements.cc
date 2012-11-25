#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/DetGroup.h"

#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

using namespace std;


namespace {
  typedef GeometricSearchDet::DetWithState DetWithState;
  inline
  void addInvalidMeas( std::vector<TrajectoryMeasurement>& result, 
		       const TrajectoryStateOnSurface& ts, const GeomDet* det) {
    result.push_back( TrajectoryMeasurement( ts, InvalidTransientRecHit::build(det, TrackingRecHit::missing), 0.F,0));
  }
  
  
  
  /** The std::vector<DetWithState> passed to this method should not be empty.
   *  In case of no compatible dets the result should be either an empty container if 
   *  the det is itself incompatible, or a container with one invalid measurement
   *  on the det surface. The method does not have enough information to do
   *  this efficiently, so it should be done by the caller, or an exception will
   *  be thrown (DetLogicError).
   */
  inline
  std::vector<TrajectoryMeasurement> 
  get(const MeasurementDetSystem* theDetSystem, 
      const DetLayer& layer,
      const std::vector<DetWithState>& compatDets,
      const TrajectoryStateOnSurface& ts, 
      const Propagator& prop, 
      const MeasurementEstimator& est) {
    std::vector<TrajectoryMeasurement> result;
    typedef TrajectoryMeasurement      TM;

    for ( auto const & ds : compatDets) {
      const MeasurementDet* mdet = theDetSystem->idToDet(ds.first->geographicalId());
      if (mdet == 0) {
	throw MeasurementDetException( "MeasurementDet not found");
      }
      
      std::vector<TM> && tmp = mdet->fastMeasurements( ds.second, ts, prop, est);
     if ( !tmp.empty()) {
       // only collect valid RecHits
       if(tmp.back().recHit()->getType() == TrackingRecHit::missing) tmp.pop_back();
       result.insert( result.end(),std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));
     }
    }
    // WARNING: we might end up with more than one invalid hit of type 'inactive' in result
    // to be fixed in order to avoid usless double traj candidates.
    
    // sort the final result
    if ( result.size() > 1) {
      sort( result.begin(), result.end(), TrajMeasLessEstim());
    }
    
    
    if ( !result.empty()) {
      // invalidMeas on Det of most compatible hit
      addInvalidMeas( result, result.front().predictedState(), result.front().recHit()->det());
    }
    else {
      // invalid state on first compatible Det
      addInvalidMeas( result, compatDets.front().second, compatDets.front().first);
    }
  
    for (auto & tm : result) tm.setLayer(&layer);
    return result;
  }

}
 
vector<TrajectoryMeasurement>
LayerMeasurements::measurements( const DetLayer& layer, 
				 const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop, 
				 const MeasurementEstimator& est) const {
  typedef DetLayer::DetWithState   DetWithState;
  vector<DetWithState> const & compatDets = layer.compatibleDets( startingState, prop, est);
  
  if (compatDets.empty()) {
    vector<TrajectoryMeasurement> result;
    pair<bool, TrajectoryStateOnSurface> compat = layer.compatible( startingState, prop, est);
    
    if ( compat.first) {
      result.push_back( TrajectoryMeasurement( compat.second, 
					       InvalidTransientRecHit::build(0, TrackingRecHit::inactive,&layer), 0.F,
					       &layer));
      LogDebug("LayerMeasurements")<<"adding a missing hit.";
    }else LogDebug("LayerMeasurements")<<"adding not measurement.";
    return result;
  }
  
  return get(theDetSystem, layer, compatDets, startingState, prop, est);
  
}


vector<TrajectoryMeasurementGroup>
LayerMeasurements::groupedMeasurements( const DetLayer& layer, 
					const TrajectoryStateOnSurface& startingState,
					const Propagator& prop, 
					const MeasurementEstimator& est) const {
  vector<TrajectoryMeasurementGroup> result;
  
  vector<DetGroup> && groups = layer.groupedCompatibleDets( startingState, prop, est);
  result.reserve(groups.size());

  tracking::TempMeasurements tmps;
  for (auto&  grp : groups) {
    if ( grp.empty() )  continue;
    
    vector<TrajectoryMeasurement> tmpVec;
    for (auto const & det : grp) {
      const MeasurementDet* mdet = theDetSystem->idToDet(det.det()->geographicalId());
      if (mdet == 0) {
	throw MeasurementDetException( "MeasurementDet not found");
      }      
      if (mdet->measurements( det.trajectoryState(), est,tmps))
	for (std::size_t i=0; i!=tmps.size(); ++i)
	  tmpVec.emplace_back(det.trajectoryState(),std::move(tmps.hits[i]),tmps.distances[i],&layer);
      tmps.clear();
    }
    
    // sort the final result
    sort( tmpVec.begin(), tmpVec.end(), TrajMeasLessEstim());
    addInvalidMeas( tmpVec, grp,layer); 
    result.emplace_back(std::move(tmpVec), std::move(grp));
  }


  // if the result is empty check if the layer is compatible (for invalid measurement)
  if (result.empty()) {
    pair<bool, TrajectoryStateOnSurface> compat = layer.compatible( startingState, prop, est);
    if ( compat.first) {
      TrajectoryMeasurement inval( compat.second, InvalidTransientRecHit::build(0, TrackingRecHit::inactive,&layer), 0.F,&layer);
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
    measVec.emplace_back(measVec.front().predictedState(), 
			 InvalidTransientRecHit::build(measVec.front().recHit()->det(), TrackingRecHit::missing),
			 0.,&layer);
  }
  else if (!group.empty()) {
    // invalid state on first compatible Det
    measVec.emplace_back(group.front().trajectoryState(), 
			 InvalidTransientRecHit::build(group.front().det(), TrackingRecHit::missing), 0.,&layer);
  }
}
