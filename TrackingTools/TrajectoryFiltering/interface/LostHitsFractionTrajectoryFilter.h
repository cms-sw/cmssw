#ifndef LostHitsFractionTrajectoryFilter_H
#define LostHitsFractionTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class LostHitsFractionTrajectoryFilter : public TrajectoryFilter {
public:

  explicit LostHitsFractionTrajectoryFilter( float maxLostHitsFraction=1./6,int transition=7 ): 
  theMaxLostHitsFraction( maxLostHitsFraction), 
  theTransition( transition) {}
  
  explicit LostHitsFractionTrajectoryFilter( const edm::ParameterSet & pset){
    theMaxLostHitsFraction = pset.existsAs<double>("maxLostHitsFraction") ? pset.getParameter<double>("maxLostHitsFraction") : 999; 
    theTransition =  pset.existsAs<int>("transition") ? pset.getParameter<int>("transition") : 0; 
  }

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const{ return TBC<Trajectory>(traj);}

  virtual std::string name() const{return "LostHitsFractionTrajectoryFilter";}

protected:

  template<class T> bool TBC(const T& traj) const {
    if( (traj.foundHits()>=theTransition &&  traj.lostHits() <= theMaxLostHitsFraction*traj.foundHits() ) ||
	(traj.foundHits()<theTransition && traj.lostHits() <= 1) ) return true;
    else
      return false;
  }

  float theMaxLostHitsFraction;
  float theTransition;

};

#endif
