#ifndef _CR_VOLUMEMULTIPLESCATTERINGESTIMATOR_H_
#define _CR_VOLUMEMULTIPLESCATTERINGESTIMATOR_H_

/** \class VolumeMultipleScatteringEstimator
 *  Estimation of multiple scattering for a finite step size in a volume.
 *  Based on path length and medium properties; neglects "higher order effects" 
 *  like magnetic field, orientation of the exit surface, etc.
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsEstimator.h"

class VolumeMaterialEffectsEstimate;
class VolumeMediumProperties;

class VolumeMultipleScatteringEstimator GCC11_FINAL : public VolumeMaterialEffectsEstimator
{  
public:
  /// Constructor with explicit mass hypothesis
  VolumeMultipleScatteringEstimator ( float mass ) :
    VolumeMaterialEffectsEstimator(mass) {}
  
  virtual ~VolumeMultipleScatteringEstimator () {}

  /// Creates an estimate
  virtual VolumeMaterialEffectsEstimate estimate (const TrajectoryStateOnSurface refTSOS,
						  double pathLength,
						  const VolumeMediumProperties& medium) const;

  virtual VolumeMultipleScatteringEstimator* clone()  const;
};

#endif
