#include "TrackingTools/PatternTools/interface/trajectoryStateClosestToBeamLine.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

bool trajectoryStateClosestToBeamLine(const Trajectory& traj,
                                      const reco::BeamSpot& bs,
                                      const Propagator* propagator,
                                      TrajectoryStateClosestToBeamLine& tscbl) {
  TrajectoryStateOnSurface stateOnSurface =
      traj.closestMeasurement(GlobalPoint(bs.x0(), bs.y0(), bs.z0())).updatedState();

  if (!stateOnSurface.isValid()) {
    edm::LogError("CannotPropagateToBeamLine") << "the state on the closest measurement is not valid. skipping track.";
    return false;
  }

  TSCBLBuilderWithPropagator builder(*propagator);
  tscbl = builder(*stateOnSurface.freeState(), bs);
  return tscbl.isValid();
}
