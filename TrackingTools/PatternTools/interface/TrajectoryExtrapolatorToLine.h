#ifndef TrajectoryExtrapolatorToLine_h_
#define TrajectoryExtrapolatorToLine_h_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Line.h"

class TrajectoryExtrapolatorToLine {

public:

  /// extrapolation with user-supplied propagator
  TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState& fts,
				       const Line& L,
				       const Propagator& p) const;

};

#endif
