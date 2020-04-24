#ifndef MaxHitsTrajectoryFilter_H
#define MaxHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class MaxHitsTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit MaxHitsTrajectoryFilter( int maxHits=10000): theMaxHits( maxHits) {}
    
  explicit MaxHitsTrajectoryFilter(const edm::ParameterSet & pset, edm::ConsumesCollector& iC):
    theMaxHits( pset.getParameter<int>("maxNumberOfHits")) {if (theMaxHits<0) theMaxHits=10000;  }

  bool qualityFilter( const Trajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }
  bool qualityFilter( const TempTrajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }

  bool toBeContinued( TempTrajectory& traj) const override {return TBC<TempTrajectory>(traj);}
  bool toBeContinued( Trajectory& traj) const override { return TBC<Trajectory>(traj);}

  std::string name() const override {return "MaxHitsTrajectoryFilter";}

 protected:

  template<class T> bool TBC(T & traj) const{
    bool ret = traj.foundHits() < theMaxHits ;
    if (!ret) traj.setStopReason(StopReason::MAX_HITS);
    return ret;
  }

  int theMaxHits;

};

#endif
