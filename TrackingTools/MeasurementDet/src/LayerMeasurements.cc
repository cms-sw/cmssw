#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/MeasurementDet/interface/CompositeDetMeasurements.h"

std::vector<TrajectoryMeasurement>
LayerMeasurements::measurements( const GeometricSearchDet& layer, 
				 const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop, 
				 const MeasurementEstimator& est) const
{
  typedef GeometricSearchDet::DetWithState   DetWithState;
  vector<DetWithState> compatDets = layer.compatibleDets( startingState, prop, est);

  vector<TrajectoryMeasurement> result;
  if (compatDets.empty()) {
    pair<bool, TrajectoryStateOnSurface> compat =
      layer.compatible( startingState, prop, est);

    if ( compat.first) {
      result.push_back( TrajectoryMeasurement( compat.second, 
					       0, 0.F));
    }
    return result;
  }
  /*
  GeometricSearchDetMeasurements cdm;
  return cdm.get( layer, compatDets, startingState, prop, est);
  */
  return std::vector<TrajectoryMeasurement>();
}
