#ifndef _TRACKER_KFUPDATOR_H_
#define _TRACKER_KFUPDATOR_H_

/** \class KFUpdator
 * Update trajectory state by combining predicted state and measurement 
 * as prescribed in the Kalman Filter algorithm 
 * (see R. Fruhwirth, NIM A262 (1987) 444). <BR>
 *
 * x_filtered = x_predicted + K * (measurement - H * x_predicted) <BR> 
 *
 * x_filtered, x_predicted    filtered and predicted state vectors <BR>
 * measurement                measurement vector <BR>
 * H "measurement matrix"     projects state vector onto measurement space <BR>
 * K                          Kalman gain matrix <BR>
 * (formulae for K and error matrix of filtered state not shown) <BR>
 *
 * This implementation works for measurements of all dimensions.
 * It relies on CLHEP double precision vectors and matrices for 
 * matrix calculations. <BR>
 *
 * Arguments: TrajectoryState &   predicted state <BR>
 *            RecHit &            reconstructed hit <BR>
 *
 * Initial author: P.Vanlaer 25.02.1999
 * Ported from ORCA.
 *
 *  \author vanlaer, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

class KFUpdator GCC11_FINAL : public TrajectoryStateUpdator {

public:

  // methods of Updator

  KFUpdator() {}

  TrajectoryStateOnSurface update(const TrajectoryStateOnSurface&,
                                  const TrackingRecHit&) const;

  template <unsigned int D> TrajectoryStateOnSurface update(const TrajectoryStateOnSurface&,
                                  const TrackingRecHit&) const;

  virtual KFUpdator * clone() const {
    return new KFUpdator(*this);
  }
};

#endif
