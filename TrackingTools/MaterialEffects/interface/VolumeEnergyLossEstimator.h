#ifndef _CR_VOLUMEENERGYLOSSESTIMATOR_H_
#define _CR_VOLUMEENERGYLOSSESTIMATOR_H_

/** \class VolumeEnergyLossEstimator
 *  Estimation of energy loss for a finite step size in a volume.
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsEstimator.h"
#include "FWCore/Utilities/interface/Visibility.h"

class VolumeMaterialEffectsEstimate;
class VolumeMediumProperties;

class VolumeEnergyLossEstimator GCC11_FINAL : public VolumeMaterialEffectsEstimator
{
public:
  /// Constructor with explicit mass hypothesis
  VolumeEnergyLossEstimator ( float mass ) :
    VolumeMaterialEffectsEstimator(mass) {}
  
  virtual ~VolumeEnergyLossEstimator () {}

  /// Creates an estimate
  virtual VolumeMaterialEffectsEstimate estimate (const TrajectoryStateOnSurface refTSOS,
						  double pathLength,
						  const VolumeMediumProperties& medium) const;

  virtual VolumeEnergyLossEstimator* clone()  const;

private:
  void computeBetheBloch (const LocalVector& localP, double pathLength,
			  const VolumeMediumProperties& medium,
			  double& deltaP, double& deltaCov00) const dso_internal;
  void computeElectrons (const LocalVector& localP, double pathLength,
			 const VolumeMediumProperties& medium,
			 double& deltaP, double& deltaCov00) const dso_internal;
};

#endif
