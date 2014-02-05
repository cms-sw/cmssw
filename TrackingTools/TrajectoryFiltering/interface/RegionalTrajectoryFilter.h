#ifndef RegionalTrajectoryFilter_h
#define RegionalTrajectoryFilter_h

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/TrajectoryFiltering/interface/MinPtTrajectoryFilter.h"

/** TrajectoryFilter checking compatibility with (the
 *  pt cut of) a tracking region. 
 */

class RegionalTrajectoryFilter : public TrajectoryFilter {
public:
  /// constructor from TrackingRegion
  explicit RegionalTrajectoryFilter (const edm::ParameterSet &  pset, edm::ConsumesCollector& iC);
  explicit RegionalTrajectoryFilter( const TrackingRegion& region);

  virtual bool qualityFilter(const TempTrajectory& traj) const;
  virtual bool qualityFilter(const Trajectory& traj) const;
    
  virtual bool toBeContinued (TempTrajectory& traj) const;
  virtual bool toBeContinued(Trajectory& traj) const;
  
  /// name method imposed by TrajectoryFilter
    std::string name () const;
  
 protected:
  const MinPtTrajectoryFilter thePtFilter;
};
#endif

