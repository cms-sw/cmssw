#ifndef AnalyticalTrajectoryExtrapolatorToLine_h_
#define AnalyticalTrajectoryExtrapolatorToLine_h_

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/CommonDetAlgo/interface/DeepCopyPointerByClone.h"

class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class Line;
class IterativeHelixExtrapolatorToLine;
class MagneticField;

/** Extrapolate to the closest approach w.r.t. a line. This class is 
 *  faster than the TrajectoryExtrapolatorToLine. The helix model is 
 *  explicitely used in the determination of the target surface.
 *  This target surface is centered on the point of closest approach
 *  on the line.
 *  The axes of the local coordinate system (x_loc, y_loc, z_loc) are 
 *  z_loc // trajectory direction at point of closest approach;
 *  x_loc normal to trajectory and along impact vector (line->helix);
 *  y_loc forms a right-handed system with the other axes.
 */

class AnalyticalTrajectoryExtrapolatorToLine {

public:
  /// constructor with default geometrical propagator
  AnalyticalTrajectoryExtrapolatorToLine (const MagneticField* field);

  /// constructor with alternative propagator
  AnalyticalTrajectoryExtrapolatorToLine (const Propagator&);

  /// extrapolation from FreeTrajectoryState
  TrajectoryStateOnSurface extrapolate (const FreeTrajectoryState& fts,
					const Line & L) const;

  /// extrapolation from TrajectoryStateOnSurface
  TrajectoryStateOnSurface extrapolate (const TrajectoryStateOnSurface tsos,
					const Line & L) const;

private:
  /// extrapolation of (multi) TSOS
  TrajectoryStateOnSurface extrapolateFullState (const TrajectoryStateOnSurface tsos, 
						 const Line& line) const;
  /// extrapolation of (single) FTS
  TrajectoryStateOnSurface extrapolateSingleState (const FreeTrajectoryState& fts, 
						   const Line& line) const;
  /// the actual propagation to a new point & momentum vector
  bool propagateWithHelix (const IterativeHelixExtrapolatorToLine& extrapolator, 
			   const Line& line,
			   GlobalPoint& x, GlobalVector& p, double& s) const;

private:
  DeepCopyPointerByClone<Propagator> thePropagator;
};

#endif




