#ifndef TrajectoryCleaning_TrajectoryCleaner_h
#define TrajectoryCleaning_TrajectoryCleaner_h

#include "TrackingTools/PatternTools/interface/Trajectory.h"

/** The component of track reconstruction that resolves ambiguities 
 *  between mutually exclusive trajectories.
 */

class TrajectoryCleaner {

 public:
  typedef vector<Trajectory> 	TrajectoryContainer;
  typedef TrajectoryContainer::iterator TrajectoryIterator;

  TrajectoryCleaner(){};
  virtual ~TrajectoryCleaner(){};

  virtual void clean( TrajectoryContainer&) const = 0;

};

#endif
