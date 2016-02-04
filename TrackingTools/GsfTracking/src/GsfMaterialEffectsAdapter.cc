#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsAdapter.h"

// // currently consistent mass is only assured by common use of configurable!
// GsfMaterialEffectsAdapter::GsfMaterialEffectsAdapter() :
//   theMEUpdator(MaterialEffectsFactory().constructComponent())
// {
//   theWeights.push_back(1.);
// }

void
GsfMaterialEffectsAdapter::compute (const TrajectoryStateOnSurface& aTSoS, 
				    const PropagationDirection aPropDir) const
{
  //
  // use deltaP from standard updator
  //
  theDeltaPs.clear();
  theDeltaPs.push_back(theMEUpdator->deltaP(aTSoS,aPropDir));
  //
  // use covariance matrix from standard updator
  //
  theDeltaCovs.clear();
  theDeltaCovs.push_back(theMEUpdator->deltaLocalError(aTSoS,aPropDir));
}
