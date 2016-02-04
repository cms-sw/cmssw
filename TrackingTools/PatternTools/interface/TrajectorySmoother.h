#ifndef _Tracker_TrajectorySmoother_H_
#define _Tracker_TrajectorySmoother_H_

#include "TrackingTools/PatternTools/interface/Trajectory.h"

/** Interface class for trajectory smoothers, 
 *  i.e. objects improving a Trajectory built elsewhere. 
 */


class TrajectorySmoother {
public:

  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef TrajectoryContainer::iterator TrajectoryIterator;

  virtual ~TrajectorySmoother() {}

  virtual TrajectoryContainer trajectories(const Trajectory&) const = 0;

  virtual TrajectorySmoother* clone() const = 0;
};

#endif
