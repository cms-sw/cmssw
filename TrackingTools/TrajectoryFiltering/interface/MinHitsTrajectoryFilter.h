#ifndef MinHitsTrajectoryFilter_H
#define MinHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"


/** A TrajectoryFilter that stops reconstruction if P_t drops
 *  below some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the minimal P_t cut.
 */

class MinHitsTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit MinHitsTrajectoryFilter( int minHits=-1):theMinHits( minHits) {}

  explicit MinHitsTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC): theMinHits( pset.getParameter<int>("minimumNumberOfHits")) {}
    
  virtual bool qualityFilter( const Trajectory& traj) const { return QF<Trajectory>(traj);}
  virtual bool qualityFilter( const TempTrajectory& traj) const { return QF<TempTrajectory>(traj);}

  virtual bool toBeContinued( TempTrajectory&) const { return TrajectoryFilter::toBeContinuedIfNotContributing ;}
  virtual bool toBeContinued( Trajectory&) const { return TrajectoryFilter::toBeContinuedIfNotContributing ;}

  virtual std::string name() const {return "MinHitsTrajectoryFilter";}

protected:

  template<class T> bool QF(const T & traj) const{
    return (traj.foundHits() >= theMinHits);
  }

  int theMinHits;

};

#endif
