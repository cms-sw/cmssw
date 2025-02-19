#ifndef MaxLostHitsTrajectoryFilter_H
#define MaxLostHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class MaxLostHitsTrajectoryFilter : public TrajectoryFilter {
public:

  explicit MaxLostHitsTrajectoryFilter( int maxHits=-1): theMaxLostHits( maxHits) {}
  
  explicit MaxLostHitsTrajectoryFilter( const edm::ParameterSet & pset):
    theMaxLostHits( pset.getParameter<int>("maxLostHits")) {}

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const{ return TBC<Trajectory>(traj);}

  virtual std::string name() const{return "MaxLostHitsTrajectoryFilter";}

protected:

  template<class T> bool TBC(const T& traj) const {
    if (traj.lostHits() > theMaxLostHits) return false;     
    else return true;
  }

  float theMaxLostHits;

};

#endif
