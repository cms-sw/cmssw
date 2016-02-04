#ifndef TsosGaussianStateConversions_H_
#define TsosGaussianStateConversions_H_
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"

namespace GaussianStateConversions {
  MultiGaussianState<5> multiGaussianStateFromTSOS (const TrajectoryStateOnSurface tsos);
  TrajectoryStateOnSurface tsosFromMultiGaussianState (const MultiGaussianState<5>& multiState,
							  const TrajectoryStateOnSurface refTsos);
}

#endif
