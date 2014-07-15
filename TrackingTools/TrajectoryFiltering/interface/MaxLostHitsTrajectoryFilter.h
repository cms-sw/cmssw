#ifndef MaxLostHitsTrajectoryFilter_H
#define MaxLostHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class MaxLostHitsTrajectoryFilter  final : public TrajectoryFilter {
public:

  explicit MaxLostHitsTrajectoryFilter( int maxHits=0): theMaxLostHits( maxHits) {}
  
  explicit MaxLostHitsTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC):
    theMaxLostHits( pset.getParameter<int>("maxLostHits")) {}

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const{ return TBC<Trajectory>(traj);}

  virtual std::string name() const{return "MaxLostHitsTrajectoryFilter";}

protected:

  template<class T> bool TBC(const T& traj) const {
    return traj.lostHits() <= theMaxLostHits;
  }

  int theMaxLostHits;

};

#endif
