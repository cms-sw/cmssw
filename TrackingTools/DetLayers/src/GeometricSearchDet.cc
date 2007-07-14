#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

std::vector<GeometricSearchDet::DetWithState> 
GeometricSearchDet::compatibleDets( const TrajectoryStateOnSurface& startingState,
				    const Propagator& prop, 
				    const MeasurementEstimator& est) const {
  std::vector<DetWithState> result;
  compatibleDetsV( startingState, prop, est,result);
  return result;
}

std::vector<DetGroup> 
GeometricSearchDet::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
					   const Propagator& prop,
					   const MeasurementEstimator& est) const {
  std::vector<DetGroup> result;
  groupedCompatibleDetsV(startingState, prop, est,result);
  return result;
}

void
GeometricSearchDetWithGroups::compatibleDetsV( const TrajectoryStateOnSurface& startingState,
					       const Propagator& prop, 
					       const MeasurementEstimator& est,
					       std::vector<DetWithState> &result) const{
  
  // standard implementation of compatibleDets() for class which have 
  // groupedCompatibleDets implemented.
  
  std::vector<DetGroup> vectorGroups;
  groupedCompatibleDetsV(startingState,prop,est,vectorGroups);
  for(std::vector<DetGroup>::const_iterator itDG=vectorGroups.begin();
      itDG!=vectorGroups.end();itDG++){
    for(std::vector<DetGroupElement>::const_iterator itDGE=itDG->begin();
	itDGE!=itDG->end();itDGE++){
      result.push_back(DetWithState(itDGE->det(),itDGE->trajectoryState()));
    }
  }
}

