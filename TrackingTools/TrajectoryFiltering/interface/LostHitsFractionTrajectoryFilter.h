#ifndef LostHitsFractionTrajectoryFilter_H
#define LostHitsFractionTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class LostHitsFractionTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit LostHitsFractionTrajectoryFilter( float maxLostHitsFraction=999.,float constantValue=1. ): 
  //  explicit LostHitsFractionTrajectoryFilter( float maxLostHitsFraction=1./10.,float constantValue=1 ): 
  theMaxLostHitsFraction( maxLostHitsFraction), 
  theConstantValue( constantValue) {}
  
  explicit LostHitsFractionTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC){
    theMaxLostHitsFraction = pset.getParameter<double>("maxLostHitsFraction"); 
    theConstantValue       = pset.getParameter<double>("constantValueForLostHitsFractionFilter"); 
  }

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const{ return TBC<Trajectory>(traj);}

  virtual std::string name() const{return "LostHitsFractionTrajectoryFilter";}

  inline edm::ParameterSetDescription getFilledConfigurationDescription() {
    edm::ParameterSetDescription desc;
    desc.add<double>("maxLostHitsFraction",                     999.);
    desc.add<double>("constantValueForLostHitsFractionFilter",    1.);
    return desc;
  }

protected:

  template<class T> bool TBC(T& traj) const {
    bool ret = traj.lostHits() <= theConstantValue + theMaxLostHitsFraction*traj.foundHits();
    if (!ret) traj.setStopReason(StopReason::LOST_HIT_FRACTION);
    return ret;
  }

  float theMaxLostHitsFraction;
  float theConstantValue;

};

#endif
