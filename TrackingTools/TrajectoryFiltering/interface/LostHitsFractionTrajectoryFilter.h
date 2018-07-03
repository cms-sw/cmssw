#ifndef LostHitsFractionTrajectoryFilter_H
#define LostHitsFractionTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class LostHitsFractionTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit LostHitsFractionTrajectoryFilter( float maxLostHitsFraction=999.,float constantValue=1 ): 
  theMaxLostHitsFraction( maxLostHitsFraction), 
  theConstantValue( constantValue) {}
  
  explicit LostHitsFractionTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC){
    theMaxLostHitsFraction = pset.getParameter<double>("maxLostHitsFraction");
    theConstantValue       = pset.getParameter<double>("constantValueForLostHitsFractionFilter");
  }

  bool qualityFilter( const Trajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }
  bool qualityFilter( const TempTrajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }

  bool toBeContinued( TempTrajectory& traj) const override { return TBC<TempTrajectory>(traj);}
  bool toBeContinued( Trajectory& traj) const override{ return TBC<Trajectory>(traj);}

  std::string name() const override{return "LostHitsFractionTrajectoryFilter";}

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
