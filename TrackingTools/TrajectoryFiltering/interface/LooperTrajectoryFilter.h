#ifndef LooperTrajectoryFilter_H
#define LooperTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class LooperTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit LooperTrajectoryFilter( int minNumberOfHitsForLoopers=13, 
				   int minNumberOfHitsPerLoop=4,
				   int extraNumberOfHitsBeforeTheFirstLoop=4): 
  theMinNumberOfHitsForLoopers(minNumberOfHitsForLoopers), 
  theMinNumberOfHitsPerLoop(minNumberOfHitsPerLoop),
  theExtraNumberOfHitsBeforeTheFirstLoop(extraNumberOfHitsBeforeTheFirstLoop){}
  
  explicit LooperTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC){
    theMinNumberOfHitsForLoopers           = pset.getParameter<int>("minNumberOfHitsForLoopers");
    theMinNumberOfHitsPerLoop              = pset.getParameter<int>("minNumberOfHitsPerLoop");
    theExtraNumberOfHitsBeforeTheFirstLoop = pset.getParameter<int>("extraNumberOfHitsBeforeTheFirstLoop");

  }

  bool qualityFilter( const Trajectory& traj) const override { return QF<Trajectory>(traj); }
  bool qualityFilter( const TempTrajectory& traj) const override { return QF<TempTrajectory>(traj);  }

  bool toBeContinued( TempTrajectory& traj) const override { return TBC<TempTrajectory>(traj);}
  bool toBeContinued( Trajectory& traj) const override{ return TBC<Trajectory>(traj);}

  std::string name() const override{return "LooperTrajectoryFilter";}

  inline edm::ParameterSetDescription getFilledConfigurationDescription() {
    edm::ParameterSetDescription desc;
    desc.add<int>("minNumberOfHitsForLoopers",          13);
    desc.add<int>("minNumberOfHitsPerLoop",              4);
    desc.add<int>("extraNumberOfHitsBeforeTheFirstLoop", 4);
    return desc;
  }

protected:

  template<class T> bool QF(const T & traj) const{
    if ( traj.isLooper() && (traj.foundHits() < theMinNumberOfHitsForLoopers) ) return false;
    else return true;
  }


  template<class T> bool TBC(T& traj) const {
    bool ret = !(traj.isLooper() &&
                 ( (traj.nLoops()*theMinNumberOfHitsPerLoop + theExtraNumberOfHitsBeforeTheFirstLoop)>traj.foundHits()));
    if (!ret)
      traj.setStopReason(StopReason::LOOPER);
    return ret;
  }

  int theMinNumberOfHitsForLoopers;
  int theMinNumberOfHitsPerLoop;
  int theExtraNumberOfHitsBeforeTheFirstLoop;


};

#endif
