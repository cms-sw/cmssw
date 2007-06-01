#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"

#include "TrackingTools/GsfTracking/interface/TsosGaussianStateConversions.h"

TrajectoryStateOnSurface
MultiTrajectoryStateMerger::merge (const TrajectoryStateOnSurface& tsos) const
{
  if ( !tsos.isValid() )  std::cout << "Merger called with invalid state" << std::endl;
  MultiGaussianState<5> multiState(GaussianStateConversions::multiGaussianStateFromTSOS(tsos));
  MultiGaussianState<5> mergedStates = theMultiStateMerger->merge(multiState);
  return GaussianStateConversions::tsosFromMultiGaussianState(mergedStates,tsos);
}
