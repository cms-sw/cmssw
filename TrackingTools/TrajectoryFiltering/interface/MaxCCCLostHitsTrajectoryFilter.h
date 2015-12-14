#ifndef MaxCCCLostHitsTrajectoryFilter_H
#define MaxCCCLostHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

class MaxCCCLostHitsTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit MaxCCCLostHitsTrajectoryFilter (int maxCCCHits=0,
                                           float CCC_value=0) :
      theMaxCCCLostHits_(maxCCCHits),
      minGoodStripCharge_(CCC_value) {}

  explicit MaxCCCLostHitsTrajectoryFilter (
      const edm::ParameterSet & pset, edm::ConsumesCollector& iC) :
      theMaxCCCLostHits_(pset.existsAs<int>("maxCCCLostHits") ? pset.getParameter<int>("maxCCCLostHits") : 9999),
      minGoodStripCharge_(clusterChargeCut(pset, "minGoodStripCharge")) {}

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const { return TBC<Trajectory>(traj);}

  virtual std::string name() const {return "MaxCCCLostHitsTrajectoryFilter";}

protected:

  template <class T> bool TBC(T& traj) const {
    bool ret = (traj.numberOfCCCBadHits(minGoodStripCharge_) <= theMaxCCCLostHits_);
    if (!ret) traj.setStopReason(StopReason::MAX_CCC_LOST_HITS);
    return ret;
  }

  int theMaxCCCLostHits_;
  float minGoodStripCharge_;
};

#endif
