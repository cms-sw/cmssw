#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MeasurementDet/interface/GeometricSearchDetMeasurements.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

std::vector<TrajectoryMeasurement>
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
					       new InvalidTransientRecHit(0), 0.F,
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
