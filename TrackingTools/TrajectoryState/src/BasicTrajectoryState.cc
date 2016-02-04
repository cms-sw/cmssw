#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

BasicTrajectoryState::~BasicTrajectoryState(){}


std::vector<TrajectoryStateOnSurface> 
BasicTrajectoryState::components() const {
  std::vector<TrajectoryStateOnSurface> result; result.reserve(1);
  result.push_back( const_cast<BasicTrajectoryState*>(this));
  return result;
}
