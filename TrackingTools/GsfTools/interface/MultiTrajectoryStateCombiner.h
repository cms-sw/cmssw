#ifndef MultiTrajectoryStateCombiner_H
#define MultiTrajectoryStateCombiner_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

/** Class which combines a set of components of a Gaussian mixture
 *  into a single component. Given all the components of a mixture, it
 *  calculates the mean and covariance matrix of the entire mixture.
 *  This combiner class can also be used in the process of transforming a
 *  Gaussian mixture into another Gaussian mixture with a smaller number
 *  of components. The relevant formulas can be found in
 *  R. Fruhwirth, Computer Physics Communications 100 (1997), 1.
 */

class MultiTrajectoryStateCombiner {

public:
  
  MultiTrajectoryStateCombiner() {}
  ~MultiTrajectoryStateCombiner() {}

  TrajectoryStateOnSurface combine(const std::vector<TrajectoryStateOnSurface>& tsos) const;

};  

#endif
