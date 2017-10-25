#ifndef MaxConsecLostHitsTrajectoryFilter_H
#define MaxConsecLostHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class MaxConsecLostHitsTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit MaxConsecLostHitsTrajectoryFilter( int maxHits=0): theMaxConsecLostHits( maxHits) {}

  explicit MaxConsecLostHitsTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC):
    theMaxConsecLostHits( pset.getParameter<int>("maxConsecLostHits")) {}

  bool qualityFilter( const Trajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }
  bool qualityFilter( const TempTrajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }

  bool toBeContinued( TempTrajectory& traj) const override { return TBC<TempTrajectory>(traj);}
  bool toBeContinued( Trajectory& traj) const override { return TBC<Trajectory>(traj);}

  std::string name() const override {return "MaxConsecLostHitsTrajectoryFilter";}

protected:

  template <class T> bool TBC(T& traj) const{
    int consecLostHit = 0;
    const  typename T::DataContainer & tms = traj.measurements();
    typename T::DataContainer::size_type itm;
    for( itm=tms.size(); itm!=0; --itm ) {
      if (tms[itm-1].recHit()->isValid()) break;
      else if ( // FIXME: restore this:   !Trajectory::inactive(tms[itm-1].recHit()->det()) &&
	       Trajectory::lost(*tms[itm-1].recHit())) consecLostHit++;
    }
  
    bool ret = consecLostHit <= theMaxConsecLostHits;
    if (!ret) traj.setStopReason(StopReason::MAX_CONSECUTIVE_LOST_HITS);
    return ret;
    }

  int theMaxConsecLostHits;

};

#endif
