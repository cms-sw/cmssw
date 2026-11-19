#ifndef TrackingTools_PatternTools_trajectoryStateClosestToBeamLine_h
#define TrackingTools_PatternTools_trajectoryStateClosestToBeamLine_h

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"

class Propagator;

/// Returns the trajectory state at the point of closest approach to the beam line.
/// Returns false if the state on the closest measurement is invalid or the TSCBL
/// itself is invalid.
bool trajectoryStateClosestToBeamLine(const Trajectory& traj,
                                      const reco::BeamSpot& bs,
                                      const Propagator* propagator,
                                      TrajectoryStateClosestToBeamLine& tscbl);

#endif
