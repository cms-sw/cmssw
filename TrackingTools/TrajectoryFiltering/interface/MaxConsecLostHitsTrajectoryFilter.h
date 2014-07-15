#ifndef MaxConsecLostHitsTrajectoryFilter_H
#define MaxConsecLostHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class MaxConsecLostHitsTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit MaxConsecLostHitsTrajectoryFilter( int maxHits=0): theMaxConsecLostHits( maxHits) {}

  explicit MaxConsecLostHitsTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC):
    theMaxConsecLostHits( pset.getParameter<int>("maxConsecLostHits")) {}

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const { return TBC<Trajectory>(traj);}

  virtual std::string name() const {return "MaxConsecLostHitsTrajectoryFilter";}

protected:

  template <class T> bool TBC(const T& traj) const{
    int consecLostHit = 0;
    const  typename T::DataContainer & tms = traj.measurements();
    typename T::DataContainer::size_type itm;
    for( itm=tms.size(); itm!=0; --itm ) {
      if (tms[itm-1].recHit()->isValid()) break;
      else if ( // FIXME: restore this:   !Trajectory::inactive(tms[itm-1].recHit()->det()) &&
	       Trajectory::lost(*tms[itm-1].recHit())) consecLostHit++;
    }
  
    return  consecLostHit <= theMaxConsecLostHits; 

    }

  int theMaxConsecLostHits;

};

#endif
