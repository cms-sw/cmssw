#ifndef TransverseImpactPointExtrapolator_h_
#define TransverseImpactPointExtrapolator_h_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "MagneticField/Engine/interface/MagneticField.h"

class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class Plane;
template <class T> class ReferenceCountingPointer;

/** Extrapolate to impact point with respect to vtx,
 * i.e. point of closest approach to vtx in 2D.
 * The surface of the returned TrajectoryStateOnSurface 
 * is chosen centred on vtx;
 * the axes of the local coordinate system (x_loc, y_loc, z_loc) are 
 * z_loc // trajectory direction in transverse plane at impact point;
 * x_loc normal to trajectory and along impact vector (impact point - vtx);
 * y_loc forms a right-handed system with the other axes.
 */

class TransverseImpactPointExtrapolator {

public:

  /// constructor with default geometrical propagator
  TransverseImpactPointExtrapolator();
  
  /// constructor with default geometrical propagator
  TransverseImpactPointExtrapolator(const MagneticField* field);
  /// constructor with user-supplied propagator
  TransverseImpactPointExtrapolator(const Propagator& u);

  /// extrapolation with default (=geometrical) propagator
  TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState& fts,
				       const GlobalPoint& vtx) const;
  /// as above, but from TrajectoryStateOnSurface
  TrajectoryStateOnSurface extrapolate(const TrajectoryStateOnSurface tsos,
				       const GlobalPoint& vtx) const;
  
  /// extrapolation with user-supplied propagator
  TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState& fts,
				       const GlobalPoint& vtx,
				       const Propagator& u) const;
  /// as above, but from TrajectoryStateOnSurface
  TrajectoryStateOnSurface extrapolate(const TrajectoryStateOnSurface tsos,
				       const GlobalPoint& vtx,
				       const Propagator& u) const;

private:
  /// extrapolation of (multi) TSOS with (internal or user-supplied) propagator
  TrajectoryStateOnSurface doExtrapolation (const TrajectoryStateOnSurface tsos, 
					    const GlobalPoint& vtx, 
					    const Propagator& u) const;
  /// extrapolation of (single) FTS with (internal or user-supplied) propagator
  TrajectoryStateOnSurface doExtrapolation (const FreeTrajectoryState& fts, 
					    const GlobalPoint& vtx, 
					    const Propagator& u) const;
  /// computation of the TIP surface
  ReferenceCountingPointer<Plane> tipSurface (const GlobalPoint& position,
						   const GlobalVector& momentum,
						   const double& signedTransverseRadius,
						   const GlobalPoint& vtx) const;

private:
  DeepCopyPointerByClone<Propagator> thePropagator;
};

#endif
