#ifndef MaxHitsTrajectoryFilter_H
#define MaxHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class MaxHitsTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit MaxHitsTrajectoryFilter( int maxHits=10000): theMaxHits( maxHits) {}
    
  explicit MaxHitsTrajectoryFilter(const edm::ParameterSet & pset, edm::ConsumesCollector& iC):
    theMaxHits( pset.getParameter<int>("maxNumberOfHits")) {if (theMaxHits<0) theMaxHits=10000;  }

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const {return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const { return TBC<Trajectory>(traj);}

  virtual std::string name() const {return "MaxHitsTrajectoryFilter";}

 protected:

  template<class T> bool TBC(const T & traj) const{
    return traj.foundHits() < theMaxHits ;
  }

  int theMaxHits;

};

#endif
