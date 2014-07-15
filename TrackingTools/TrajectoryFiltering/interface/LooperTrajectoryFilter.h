#ifndef LooperTrajectoryFilter_H
#define LooperTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class LooperTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit LooperTrajectoryFilter( int minNumberOfHits=13, 
				   int minNumberOfHitsPerLoop=4,
				   int extraNumberOfHitsBeforeTheFirstLoop=4): 
  theMinNumberOfHits(minNumberOfHits), 
  theMinNumberOfHitsPerLoop(minNumberOfHitsPerLoop),
  theExtraNumberOfHitsBeforeTheFirstLoop(extraNumberOfHitsBeforeTheFirstLoop){}
  
  explicit LooperTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC){
    theMinNumberOfHits = pset.existsAs<int>("minNumberOfHits") ? 
      pset.getParameter<int>("minNumberOfHits") : 13; 
    theMinNumberOfHitsPerLoop= pset.existsAs<int>("minNumberOfHitsPerLoop") ? 
      pset.getParameter<int>("minNumberOfHitsPerLoop") : 4; 
    theExtraNumberOfHitsBeforeTheFirstLoop= pset.existsAs<int>("extraNumberOfHitsBeforeTheFirstLoop") ? 
      pset.getParameter<int>("extraNumberOfHitsBeforeTheFirstLoop") : 4; 

  }

  virtual bool qualityFilter( const Trajectory& traj) const { return QF<Trajectory>(traj); }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return QF<TempTrajectory>(traj);  }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const{ return TBC<Trajectory>(traj);}

  virtual std::string name() const{return "LooperTrajectoryFilter";}

protected:

  template<class T> bool QF(const T & traj) const{
    if ( traj.isLooper() && (traj.foundHits() < theMinNumberOfHits) ) return false;
    else return true;
  }


  template<class T> bool TBC(const T& traj) const {
    if(traj.isLooper() && 
       ( (traj.nLoops()*theMinNumberOfHitsPerLoop + theExtraNumberOfHitsBeforeTheFirstLoop)>traj.foundHits()) )
      return false;
    else
      return true;
  }

  int theMinNumberOfHits;
  int theMinNumberOfHitsPerLoop;
  int theExtraNumberOfHitsBeforeTheFirstLoop;


};

#endif
