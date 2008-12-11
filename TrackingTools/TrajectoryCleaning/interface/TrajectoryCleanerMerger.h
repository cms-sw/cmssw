#ifndef TrajectoryCleaning_TrajectoryCleanerMerger_h
#define TrajectoryCleaning_TrajectoryCleanerMerger_h

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

class TrajectoryCleanerMerger : public TrajectoryCleaner {

 public:
  TrajectoryCleanerMerger(){};
  virtual ~TrajectoryCleanerMerger(){};

  using TrajectoryCleaner::clean;
  virtual void clean( TrajectoryPointerContainer&) const; 

  void clean(TrajectoryContainer& trajs)const ;

 private:
  void reOrderMeasurements(Trajectory& traj) const;
  bool sameSeed (const TrajectorySeed & s1, const TrajectorySeed & s2) const;
  int getLayer(const DetId & id) const;
};

#endif
