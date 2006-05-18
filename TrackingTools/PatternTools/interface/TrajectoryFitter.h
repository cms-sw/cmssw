#ifndef _TrajectoryFitter_H_
#define _TrajectoryFitter_H_

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TrajectorySeed;
class TrajectoryStateOnSurface;

/** Interface class for trajectory fitters, i.e. computing
 *  Trajectories out of a seed and/or collection of RecHits.
 */


class TrajectoryFitter {
public:

  typedef Trajectory::RecHitContainer      RecHitContainer;

  virtual ~TrajectoryFitter() {}

  virtual vector<Trajectory> fit(const Trajectory&) const = 0;
  virtual vector<Trajectory> fit(const TrajectorySeed&,
				 const RecHitContainer&) const = 0;
  virtual vector<Trajectory> fit(const TrajectorySeed&,
				 const RecHitContainer&, 
				 const TrajectoryStateOnSurface&) const = 0;

  virtual TrajectoryFitter* clone() const = 0;
};

#endif
