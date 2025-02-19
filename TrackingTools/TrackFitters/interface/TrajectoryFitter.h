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
  enum fitType {standard=0, looper=1};

  typedef TrajectoryFitterRecord Record;
  typedef Trajectory::RecHitContainer      RecHitContainer;

  virtual ~TrajectoryFitter() {}

  virtual std::vector<Trajectory> fit(const Trajectory&) const = 0;
  virtual std::vector<Trajectory> fit(const Trajectory& traj, fitType type) const {return fit(traj);}


  virtual std::vector<Trajectory> fit(const TrajectorySeed&,
				      const RecHitContainer&) const = 0;
  virtual std::vector<Trajectory> fit(const TrajectorySeed& seed,
				      const RecHitContainer& hits, fitType type) const {return fit(seed,hits);}


  virtual std::vector<Trajectory> fit(const TrajectorySeed&,
				      const RecHitContainer&, 
				      const TrajectoryStateOnSurface&) const = 0;
  virtual std::vector<Trajectory> fit(const TrajectorySeed& seed,
				      const RecHitContainer& hits, 
				      const TrajectoryStateOnSurface& tsos,
				      fitType type) const {return fit(seed,hits,tsos);}


  virtual TrajectoryFitter* clone() const = 0;
};

#endif
