#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsAdapter.h"

// // currently consistent mass is only assured by common use of configurable!

void
GsfMaterialEffectsAdapter::compute (const TrajectoryStateOnSurface& aTSoS, 
				    const PropagationDirection aPropDir, Effect effects[]) const
{
  //
  // use deltaP and covariance matrix from standard updator
  theMEUpdator->compute(aTSoS,aPropDir, effects[0]);

}
