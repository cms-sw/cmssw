#ifndef _CR_VOLUMEMATERIALEFFECTSUPDATOR_H_
#define _CR_VOLUMEMATERIALEFFECTSUPDATOR_H_

/** \class VolumeMaterialEffectsUpdator
 *  Computes an updated TrajectoryStateOnSurface after applying
 *  estimated material effects.
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include <vector>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class VolumeMaterialEffectsEstimate;

class VolumeMaterialEffectsUpdator
{  
public:
  typedef VolumeMaterialEffectsEstimate Estimate;
  typedef std::vector<const VolumeMaterialEffectsEstimate*> EstimateContainer;

public:
  VolumeMaterialEffectsUpdator () {}

  /** Updates TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected.
   */
  TrajectoryStateOnSurface updateState (const TrajectoryStateOnSurface& TSoS, 
					const PropagationDirection propDir,
					const Estimate& estimate) const;

  /** Updates TrajectoryStateOnSurface with several material effects
   *    (momentum and covariance matrix are potentially affected.
   */
  TrajectoryStateOnSurface updateState (const TrajectoryStateOnSurface& TSoS, 
					const PropagationDirection propDir,
					const EstimateContainer& estimates) const;
  
};

#endif
