#ifndef _CR_VOLUMEMATERIALEFFECTSESTIMATOR_H_
#define _CR_VOLUMEMATERIALEFFECTSESTIMATOR_H_

/** \class VolumeMaterialEffectsEstimator
 *  Base class for the estimation of material effects in a volume
 *  (finite step size): produces a VolumeMaterialEffectsEstimate.
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class VolumeMaterialEffectsEstimate;
class VolumeMediumProperties;

class VolumeMaterialEffectsEstimator {  
public:
  /// Constructor with explicit mass hypothesis
  VolumeMaterialEffectsEstimator ( float mass ) :
    theMass(mass) {}

  virtual ~VolumeMaterialEffectsEstimator () {}

  /// Creates an estimate
  virtual VolumeMaterialEffectsEstimate estimate (const TrajectoryStateOnSurface refTSOS,
						  double pathLength,
						  const VolumeMediumProperties& medium) const = 0;
  /// Particle mass assigned at construction.
  virtual float mass () const {return theMass;}
  
  virtual VolumeMaterialEffectsEstimator* clone()  const = 0;
  
private:
  float theMass;
};

#endif
