#ifndef _TrackerReco_TrajectoryStateLessWeight_h_
#define _TrackerReco_TrajectoryStateLessWeight_h_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

/** \class TrajectoryStateLessWeight
 * Compare two TrajectoryStateOnSurface acc. to their weight.
 */

class TrajectoryStateLessWeight {
  
public:
  TrajectoryStateLessWeight() {}
  bool operator()(const TrajectoryStateOnSurface a, 
		  const TrajectoryStateOnSurface b) const
  {
    if ( !a.isValid() || !b.isValid() )  return false;
    return a.weight()>b.weight();
  }
};

#endif
