#ifndef TrajectoryCleaning_TrajectoryCleaner_h
#define TrajectoryCleaning_TrajectoryCleaner_h

#include "TrackingTools/PatternTools/interface/Trajectory.h"

/** The component of track reconstruction that resolves ambiguities 
 *  between mutually exclusive trajectories.
 */

class TrajectoryCleaner {

 public:
  typedef std::vector<Trajectory> 	TrajectoryContainer;
  typedef std::vector<Trajectory*> 	TrajectoryPointerContainer;
  typedef TrajectoryContainer::iterator TrajectoryIterator;
  typedef TrajectoryPointerContainer::iterator TrajectoryPointerIterator;

  TrajectoryCleaner(){};
  virtual ~TrajectoryCleaner(){};

  virtual void clean( TrajectoryContainer&) const;
  virtual void clean( TrajectoryPointerContainer&) const = 0;

};

#endif
