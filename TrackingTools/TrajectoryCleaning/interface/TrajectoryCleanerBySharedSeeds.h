#ifndef TrajectoryCleaning_TrajectoryCleanerBySharedSeeds_h
#define TrajectoryCleaning_TrajectoryCleanerBySharedSeeds_h

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"



class TrajectoryCleanerBySharedSeeds : public TrajectoryCleaner
{
public:
  typedef std::vector<Trajectory*> TrajectoryPointerContainer;

  TrajectoryCleanerBySharedSeeds(const edm::ParameterSet & iConfig){};
  virtual ~TrajectoryCleanerBySharedSeeds(){};

  using TrajectoryCleaner::clean;
  virtual void clean(TrajectoryPointerContainer&) const; 
  void clean                  (std::vector<Trajectory> & trajs) const; 

private:
  bool sameSeed(const TrajectorySeed & s1, const TrajectorySeed & s2) const;
};

#endif
