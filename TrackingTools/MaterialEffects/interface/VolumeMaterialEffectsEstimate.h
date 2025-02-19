#ifndef VOLUMEMATERIALEFFECTSESTIMATE_H_
#define VOLUMEMATERIALEFFECTSESTIMATE_H_

/** \class VolumeMaterialEffectsEstimate
 *  Holds an estimation of the material effects in a volume
 *  (finite step size)
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

class VolumeMaterialEffectsEstimate
{  
public:
  /** Constructor with explicit mass hypothesis
   */
  VolumeMaterialEffectsEstimate ( float deltaP, AlgebraicSymMatrix55 covariance ) :
    theDeltaP(deltaP),
    theDeltaCov(covariance) {}

  ~VolumeMaterialEffectsEstimate () {}

  /** Change in |p| from material effects for propagation 
   *  along the momentum vector */
  double deltaP () const {return theDeltaP;}

  /** Contribution to covariance matrix from material effects
   *  (in a local co-ordinate system perpendicular to the track) */
  const AlgebraicSymMatrix55& deltaLocalError () const {return theDeltaCov;}  

 private:
  const double theDeltaP;
  const AlgebraicSymMatrix55 theDeltaCov;
};

#endif
