#ifndef _TRACKER_UPDATOR_H_
#define _TRACKER_UPDATOR_H_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class TransientTrackingRecHit;
  
/** The TrajectoryState updator is a basic track fititng component 
 *  that combines the information from a measurement
 *  (a RecHit) and a predicted TrajectoryState, as in the Kalman filter formalism.
 *  The actual implementation need not be a Kalman filter (but usually is).
 */
  
class TrajectoryStateUpdator {
 public:
  
  TrajectoryStateUpdator() {}
  virtual ~TrajectoryStateUpdator() {}
  
  virtual TrajectoryStateOnSurface update(const TrajectoryStateOnSurface&,
					  const TransientTrackingRecHit&) const = 0;
  
  virtual TrajectoryStateUpdator * clone() const = 0;
  
};

#endif
