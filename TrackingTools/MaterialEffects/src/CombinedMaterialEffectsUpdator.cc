#include "TrackingTools/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"

//
// Computation: combine updates on momentum and cov. matrix from the multiple
// scattering and energy loss updators and store them in private variables
//
void CombinedMaterialEffectsUpdator::compute (const TrajectoryStateOnSurface& TSoS,
					      const PropagationDirection propDir) const
{
  theDeltaP = theMSUpdator.deltaP(TSoS,propDir) + theELUpdator.deltaP(TSoS,propDir);
  theDeltaCov = theMSUpdator.deltaLocalError(TSoS,propDir) + theELUpdator.deltaLocalError(TSoS,propDir);
}

