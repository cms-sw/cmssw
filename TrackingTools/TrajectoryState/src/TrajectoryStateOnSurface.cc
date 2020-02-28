#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"

typedef BasicSingleTrajectoryState BTSOS;

void TrajectoryStateOnSurface::update(const LocalTrajectoryParameters& p,
                                      const SurfaceType& aSurface,
                                      const MagneticField* field,
                                      const SurfaceSide side) {
  if (data().canUpdateLocalParameters()) {
    unsharedData().update(p, aSurface, field, side);
  } else {
    *this = TrajectoryStateOnSurface(p, aSurface, field, side);
  }
}

void TrajectoryStateOnSurface::update(const LocalTrajectoryParameters& p,
                                      const LocalTrajectoryError& err,
                                      const SurfaceType& aSurface,
                                      const MagneticField* field,
                                      const SurfaceSide side) {
  if (data().canUpdateLocalParameters()) {
    unsharedData().update(1., p, err, aSurface, field, side);
  } else {
    *this = TrajectoryStateOnSurface(1., p, err, aSurface, field, side);
  }
}
