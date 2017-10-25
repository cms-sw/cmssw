#ifndef _GSFMULTISTATEUPDATOR_H_
#define _GSFMULTISTATEUPDATOR_H_

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

class TrajectoryStateOnSurface;
class TrackingRecHit;

/** Class which updates a Gaussian mixture trajectory state 
 *  with the information from a
 *  reconstructed hit according to the Gaussian-sum filter (GSF) strategy.
 *  The relevant formulas can be derived from those described in
 *  R. Fruhwirth, Computer Physics Communications 100 (1997), 1.
 */

class GsfMultiStateUpdator : public TrajectoryStateUpdator {

public:

  GsfMultiStateUpdator() {}

  TrajectoryStateOnSurface update(const TrajectoryStateOnSurface&,
                                  const TrackingRecHit&) const override;

  GsfMultiStateUpdator * clone() const override {
    return new GsfMultiStateUpdator(*this);
  }
};

#endif
