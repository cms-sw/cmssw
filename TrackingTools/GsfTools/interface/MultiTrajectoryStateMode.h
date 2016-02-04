#ifndef MultiTrajectoryStateMode_H_
#define MultiTrajectoryStateMode_H_

/** Extract mode information from a TrajectoryStateOnSurface. */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

class TrajectoryStateOnSurface;

class MultiTrajectoryStateMode {
public:
  /** Cartesian momentum from 1D mode calculation in cartesian co-ordinates.
   *  Return value true for success. */
  bool momentumFromModeCartesian (const TrajectoryStateOnSurface tsos,
				  GlobalVector& momentum) const;
  /** Cartesian position from 1D mode calculation in cartesian co-ordinates.
   *  Return value true for success. */
  bool positionFromModeCartesian (const TrajectoryStateOnSurface tsos,
				  GlobalPoint& position) const;
  /** Cartesian momentum from 1D mode calculation in local co-ordinates (q/p, dx/dz, dy/dz).
   *  Return value true for success. */
  bool momentumFromModeLocal (const TrajectoryStateOnSurface tsos,
			      GlobalVector& momentum) const;
  /** Cartesian position from 1D mode calculation in local co-ordinates (x, y).
   *  Return value true for success. */
  bool positionFromModeLocal (const TrajectoryStateOnSurface tsos,
			      GlobalPoint& position) const;
  /** Momentum from 1D mode calculation in q/p. Return value true for sucess. */
  bool momentumFromModeQP (const TrajectoryStateOnSurface tsos,
			   double& momentum) const;
  /** Momentum from 1D mode calculation in p. Return value true for sucess. */
  bool momentumFromModeP (const TrajectoryStateOnSurface tsos,
			  double& momentum) const;
  /** Cartesian momentum from 1D mode calculation in p, phi, eta.
   *  Return value true for success. */
  bool momentumFromModePPhiEta (const TrajectoryStateOnSurface tsos,
				GlobalVector& momentum) const;
  /** Charge from 1D mode calculation in q/p. Q=0 in case of failure. */
  int chargeFromMode (const TrajectoryStateOnSurface tsos) const;
};


#endif

