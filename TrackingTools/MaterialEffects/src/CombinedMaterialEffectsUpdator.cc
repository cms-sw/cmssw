#include "TrackingTools/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"

//
// Computation: combine updates on momentum and cov. matrix from the multiple
// scattering and energy loss updators and store them in private variables
//
void CombinedMaterialEffectsUpdator::compute (const TrajectoryStateOnSurface& TSoS,
					      const PropagationDirection propDir, Effect & effect) const
{
  theMSUpdator.compute(TSoS,propDir,effect);
  theELUpdator.compute(TSoS,propDir,effect);
}

