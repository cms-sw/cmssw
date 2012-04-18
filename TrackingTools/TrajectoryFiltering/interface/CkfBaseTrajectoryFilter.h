#ifndef CkfBaseTrajectoryFilter_H
#define CkfBaseTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "TrackingTools/TrajectoryFiltering/interface/ChargeSignificanceTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxConsecLostHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxLostHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MinHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MinPtTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/LostHitsFractionTrajectoryFilter.h"


class CkfBaseTrajectoryFilter : public TrajectoryFilter {
public:

  explicit CkfBaseTrajectoryFilter( const edm::ParameterSet & pset){
    //define the filters by default in the BaseCkfTrajectoryBuilder
    theChargeSignificanceTrajectoryFilter =  new ChargeSignificanceTrajectoryFilter(pset);
    theMaxLostHitsTrajectoryFilter = new MaxLostHitsTrajectoryFilter(pset);
    theMaxConsecLostHitsTrajectoryFilter = new MaxConsecLostHitsTrajectoryFilter(pset);
    theMinPtTrajectoryFilter = new MinPtTrajectoryFilter(pset);
    theMaxHitsTrajectoryFilter = new MaxHitsTrajectoryFilter(pset);
    theMinHitsTrajectoryFilter = new MinHitsTrajectoryFilter(pset);
    theLostHitsFractionTrajectoryFilter = new LostHitsFractionTrajectoryFilter(pset);
  }
  
  virtual bool qualityFilter( const Trajectory& traj) const {return QF<Trajectory>(traj);}
  virtual bool qualityFilter( const TempTrajectory& traj) const {return QF<TempTrajectory>(traj);}
 
  virtual bool toBeContinued( Trajectory& traj) const {return TBC<Trajectory>(traj);}
  virtual bool toBeContinued( TempTrajectory& traj) const {return TBC<TempTrajectory>(traj);}

  virtual  std::string name() const { return "CkfBaseTrajectoryFilter";}
  
protected:

  template <class T> bool QF(const T& traj) const{
    if (!theChargeSignificanceTrajectoryFilter->qualityFilter(traj)) return false;            
    if (!theMinHitsTrajectoryFilter->qualityFilter(traj)) return false;
    if (!theMinPtTrajectoryFilter->qualityFilter(traj)) return false;
    return true;}

  template <class T> bool TBC(T& traj) const{
    if (!theMaxHitsTrajectoryFilter->toBeContinued(traj)) return false;     
    if (!theMaxLostHitsTrajectoryFilter->toBeContinued(traj)) return false;
    if (!theMaxConsecLostHitsTrajectoryFilter->toBeContinued(traj)) return false;
    if (!theLostHitsFractionTrajectoryFilter->toBeContinued(traj)) return false;
    if (!theMinPtTrajectoryFilter->toBeContinued(traj)) return false;     
    if (!theChargeSignificanceTrajectoryFilter->toBeContinued(traj)) return false;
    return true;}

  

  ChargeSignificanceTrajectoryFilter * theChargeSignificanceTrajectoryFilter;
  MaxConsecLostHitsTrajectoryFilter * theMaxConsecLostHitsTrajectoryFilter;
  MaxHitsTrajectoryFilter * theMaxHitsTrajectoryFilter;
  MaxLostHitsTrajectoryFilter * theMaxLostHitsTrajectoryFilter;
  LostHitsFractionTrajectoryFilter * theLostHitsFractionTrajectoryFilter;
  MinHitsTrajectoryFilter * theMinHitsTrajectoryFilter;
  MinPtTrajectoryFilter * theMinPtTrajectoryFilter;
};

#endif
