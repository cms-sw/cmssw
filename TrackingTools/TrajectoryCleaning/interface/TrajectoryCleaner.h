#ifndef TrajectoryCleaning_TrajectoryCleaner_h
#define TrajectoryCleaning_TrajectoryCleaner_h

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** The component of track reconstruction that resolves ambiguities 
 *  between mutually exclusive trajectories.
 */

class TrackingComponentsRecord;

class TrajectoryCleaner {

 public:
  typedef TrackingComponentsRecord      Record;
  typedef std::vector<Trajectory> 	TrajectoryContainer;
  typedef std::vector<Trajectory*> 	TrajectoryPointerContainer;
  typedef TrajectoryContainer::iterator TrajectoryIterator;
  typedef TrajectoryPointerContainer::iterator TrajectoryPointerIterator;

  TrajectoryCleaner(){};
  TrajectoryCleaner(edm::ParameterSet & iConfig){};
  virtual ~TrajectoryCleaner(){};

  virtual void clean( TrajectoryContainer&) const;
  virtual void clean( TrajectoryPointerContainer&) const = 0;

};

#endif
