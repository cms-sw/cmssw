#ifndef AnalyticalImpactPointExtrapolator_h_
#define AnalyticalImpactPointExtrapolator_h_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class IterativeHelixExtrapolatorToLine;
class MagneticField; 

/** Extrapolate to impact point with respect to vtx, i.e. to the point 
 *  of closest approach to vtx in 3D. It is slightly faster than the 
 *  ImpactPointExtrapolator. The helix model is explicitely used in
 *  the determination of the target surface.
 *  This target surface is centered on vtx;
 *  the axes of the local coordinate system (x_loc, y_loc, z_loc) are 
 *  z_loc // trajectory direction at impact point;
 *  x_loc normal to trajectory and along impact vector (impact point - vtx);
 *  y_loc forms a right-handed system with the other axes.
 */

class AnalyticalImpactPointExtrapolator {

public:

  /// constructor with default geometrical propagator
  AnalyticalImpactPointExtrapolator ( const MagneticField* field);

  /// constructor with alternative propagator
  AnalyticalImpactPointExtrapolator (const Propagator&, const MagneticField*);

  /// extrapolation from FreeTrajectoryState
  TrajectoryStateOnSurface extrapolate (const FreeTrajectoryState& fts, 
					const GlobalPoint& vtx) const;
  /// as above, but from TrajectoryStateOnSurface
  TrajectoryStateOnSurface extrapolate (const TrajectoryStateOnSurface tsos, 
				        const GlobalPoint& vtx) const;

private:
  /// extrapolation of (multi) TSOS
  TrajectoryStateOnSurface extrapolateFullState(const TrajectoryStateOnSurface tsos, 
						const GlobalPoint& vertex) const;
  /// extrapolation of (single) FTS
  TrajectoryStateOnSurface extrapolateSingleState(const FreeTrajectoryState& fts, 
						  const GlobalPoint& vertex) const;
  /// the actual propagation to a new point & momentum vector
  bool propagateWithHelix (const IterativeHelixExtrapolatorToLine& extrapolator,
			   const GlobalPoint& vertex,
			   GlobalPoint& x, GlobalVector& p, double& s) const;

private:
  DeepCopyPointerByClone<Propagator> thePropagator;
  const MagneticField* theField;
};

#endif
