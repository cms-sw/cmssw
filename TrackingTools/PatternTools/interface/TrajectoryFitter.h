#ifndef _TrajectoryFitter_H_
#define _TrajectoryFitter_H_

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TrajectorySeed;
class TrajectoryStateOnSurface;

/** Interface class for trajectory fitters, i.e. computing
 *  Trajectories out of a seed and/or collection of RecHits.
 */

#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"

class TrajectoryFitter {
public:
  
  typedef TrajectoryFitterRecord Record;
  typedef Trajectory::RecHitContainer      RecHitContainer;

  virtual ~TrajectoryFitter() {}

  virtual std::vector<Trajectory> fit(const Trajectory&) const = 0;
  virtual std::vector<Trajectory> fit(const TrajectorySeed&,
				      const RecHitContainer&) const = 0;
  virtual std::vector<Trajectory> fit(const TrajectorySeed&,
				      const RecHitContainer&, 
				      const TrajectoryStateOnSurface&) const = 0;

  virtual TrajectoryFitter* clone() const = 0;
};

#endif
