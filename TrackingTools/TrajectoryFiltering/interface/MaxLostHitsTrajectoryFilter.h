#ifndef MaxLostHitsTrajectoryFilter_H
#define MaxLostHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class MaxLostHitsTrajectoryFilter  final : public TrajectoryFilter {
public:

  explicit MaxLostHitsTrajectoryFilter( int maxHits=0): theMaxLostHits( maxHits) {}
  
  explicit MaxLostHitsTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC):
    theMaxLostHits( pset.getParameter<int>("maxLostHits")) {}

  bool qualityFilter( const Trajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }
  bool qualityFilter( const TempTrajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }

  bool toBeContinued( TempTrajectory& traj) const override { return TBC<TempTrajectory>(traj);}
  bool toBeContinued( Trajectory& traj) const override{ return TBC<Trajectory>(traj);}

  std::string name() const override{return "MaxLostHitsTrajectoryFilter";}

protected:

  template<class T> bool TBC(T& traj) const {
    bool ret = traj.lostHits() <= theMaxLostHits;
    if (!ret) traj.setStopReason(StopReason::MAX_LOST_HITS);
    return ret;
  }

  int theMaxLostHits;

};

#endif
