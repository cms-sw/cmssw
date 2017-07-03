#ifndef TrajectoryCleaning_TrajectoryCleanerBySharedSeeds_h
#define TrajectoryCleaning_TrajectoryCleanerBySharedSeeds_h

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"



class TrajectoryCleanerBySharedSeeds : public TrajectoryCleaner
{
public:
  typedef std::vector<Trajectory*> TrajectoryPointerContainer;

  TrajectoryCleanerBySharedSeeds(const edm::ParameterSet & iConfig){};
  ~TrajectoryCleanerBySharedSeeds() override{};

  using TrajectoryCleaner::clean;
  void clean(TrajectoryPointerContainer&) const override; 
  void clean                  (std::vector<Trajectory> & trajs) const override; 

private:
  bool sameSeed(const TrajectorySeed & s1, const TrajectorySeed & s2) const;
};

#endif
