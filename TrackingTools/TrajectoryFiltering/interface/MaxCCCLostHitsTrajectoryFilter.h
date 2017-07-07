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
      theMaxCCCLostHits_(pset.getParameter<int>("maxCCCLostHits")),
      minGoodStripCharge_(clusterChargeCut(pset, "minGoodStripCharge")) {}

  bool qualityFilter( const Trajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }
  bool qualityFilter( const TempTrajectory& traj) const override { return TrajectoryFilter::qualityFilterIfNotContributing; }

  bool toBeContinued( TempTrajectory& traj) const override { return TBC<TempTrajectory>(traj);}
  bool toBeContinued( Trajectory& traj) const override { return TBC<Trajectory>(traj);}

  std::string name() const override {return "MaxCCCLostHitsTrajectoryFilter";}

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
