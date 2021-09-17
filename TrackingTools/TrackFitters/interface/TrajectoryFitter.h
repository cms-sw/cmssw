#ifndef _TrajectoryFitter_H_
#define _TrajectoryFitter_H_

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <memory>

class TrajectorySeed;
class TrajectoryStateOnSurface;
class TkCloner;

/** Interface class for trajectory fitters, i.e. computing
 *  Trajectories out of a seed and/or collection of RecHits.
 */

#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"

class TrajectoryFitter {
public:
  enum fitType { standard = 0, looper = 1 };

  typedef TrajectoryFitterRecord Record;
  typedef Trajectory::RecHitContainer RecHitContainer;

  virtual ~TrajectoryFitter() {}
  virtual std::unique_ptr<TrajectoryFitter> clone() const = 0;

  // FIXME a prototype: final inplementaiton may differ
  virtual void setHitCloner(TkCloner const*) = 0;

  // new interface return one trajectory: if fit fails trajectory is invalid...
  virtual Trajectory fitOne(const Trajectory& traj, fitType type = standard) const = 0;
  virtual Trajectory fitOne(const TrajectorySeed& seed,
                            const RecHitContainer& hits,
                            fitType typee = standard) const = 0;
  virtual Trajectory fitOne(const TrajectorySeed& seed,
                            const RecHitContainer& hits,
                            const TrajectoryStateOnSurface& tsos,
                            fitType type = standard) const = 0;

  // backward compatible interface...
  std::vector<Trajectory> fit(const Trajectory& traj, fitType type = standard) const {
    return makeVect(fitOne(traj, type));
  }

  std::vector<Trajectory> fit(const TrajectorySeed& seed, const RecHitContainer& hits, fitType type = standard) const {
    return makeVect(fitOne(seed, hits, type));
  }
  std::vector<Trajectory> fit(const TrajectorySeed& seed,
                              const RecHitContainer& hits,
                              const TrajectoryStateOnSurface& tsos,
                              fitType type = standard) const {
    return makeVect(fitOne(seed, hits, tsos, type));
  }

private:
  static std::vector<Trajectory> makeVect(Trajectory&& outTraj) {
    if (outTraj.isValid())
      return std::vector<Trajectory>(1, outTraj);
    return std::vector<Trajectory>();
  }
};

#endif
