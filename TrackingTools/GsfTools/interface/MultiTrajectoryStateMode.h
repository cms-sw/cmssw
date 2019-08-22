#ifndef TrackingTools_GsfTools_MultiTrajectoryStateMode_h
#define TrackingTools_GsfTools_MultiTrajectoryStateMode_h

/** Extract mode information from a TrajectoryStateOnSurface. */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

class TrajectoryStateOnSurface;

namespace multiTrajectoryStateMode {
  /** Cartesian momentum from 1D mode calculation in cartesian co-ordinates.
   *  Return value true for success. */
  bool momentumFromModeCartesian(TrajectoryStateOnSurface const& tsos, GlobalVector& momentum);
  /** Cartesian position from 1D mode calculation in cartesian co-ordinates.
   *  Return value true for success. */
  bool positionFromModeCartesian(TrajectoryStateOnSurface const& tsos, GlobalPoint& position);
  /** Cartesian momentum from 1D mode calculation in local co-ordinates (q/p, dx/dz, dy/dz).
   *  Return value true for success. */
  bool momentumFromModeLocal(TrajectoryStateOnSurface const& tsos, GlobalVector& momentum);
  /** Cartesian position from 1D mode calculation in local co-ordinates (x, y).
   *  Return value true for success. */
  bool positionFromModeLocal(TrajectoryStateOnSurface const& tsos, GlobalPoint& position);
  /** Momentum from 1D mode calculation in q/p. Return value true for sucess. */
  bool momentumFromModeQP(TrajectoryStateOnSurface const& tsos, double& momentum);
  /** Momentum from 1D mode calculation in p. Return value true for sucess. */
  bool momentumFromModeP(TrajectoryStateOnSurface const& tsos, double& momentum);
  /** Cartesian momentum from 1D mode calculation in p, phi, eta.
   *  Return value true for success. */
  bool momentumFromModePPhiEta(TrajectoryStateOnSurface const& tsos, GlobalVector& momentum);
  /** Charge from 1D mode calculation in q/p. Q=0 in case of failure. */
  int chargeFromMode(TrajectoryStateOnSurface const& tsos);
};  // namespace multiTrajectoryStateMode

#endif
