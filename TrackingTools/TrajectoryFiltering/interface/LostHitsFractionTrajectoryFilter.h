#ifndef LostHitsFractionTrajectoryFilter_H
#define LostHitsFractionTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class LostHitsFractionTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit LostHitsFractionTrajectoryFilter( float maxLostHitsFraction=1./10.,float constantValue=1 ): 
  theMaxLostHitsFraction( maxLostHitsFraction), 
  theConstantValue( constantValue) {}
  
  explicit LostHitsFractionTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC){
    theMaxLostHitsFraction = pset.existsAs<double>("maxLostHitsFraction") ? 
      pset.getParameter<double>("maxLostHitsFraction") : 999; 
    theConstantValue =  pset.existsAs<double>("constantValueForLostHitsFractionFilter") ? 
      pset.getParameter<double>("constantValueForLostHitsFractionFilter") : 1; 
  }

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const{ return TBC<Trajectory>(traj);}

  virtual std::string name() const{return "LostHitsFractionTrajectoryFilter";}

protected:

  template<class T> bool TBC(const T& traj) const {
    return traj.lostHits() <= theConstantValue + theMaxLostHitsFraction*traj.foundHits();
  }

  float theMaxLostHitsFraction;
  float theConstantValue;

};

#endif
