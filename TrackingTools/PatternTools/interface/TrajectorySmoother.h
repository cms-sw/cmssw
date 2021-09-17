#ifndef _Tracker_TrajectorySmoother_H_
#define _Tracker_TrajectorySmoother_H_

#include "TrackingTools/PatternTools/interface/Trajectory.h"

class TkCloner;

/** Interface class for trajectory smoothers, 
 *  i.e. objects improving a Trajectory built elsewhere. 
 */
class TrajectorySmoother {
public:
  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef TrajectoryContainer::iterator TrajectoryIterator;

  virtual ~TrajectorySmoother() {}

  virtual TrajectoryContainer trajectories(const Trajectory& traj) const {
    Trajectory&& nt = trajectory(traj);
    if (nt.isValid())
      return TrajectoryContainer(1, nt);
    return TrajectoryContainer();
  }
  virtual Trajectory trajectory(const Trajectory&) const = 0;

  virtual TrajectorySmoother* clone() const = 0;
  // FIXME a prototype: final inplementaiton may differ
  virtual void setHitCloner(TkCloner const*) = 0;
};

#endif
