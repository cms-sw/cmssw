#ifndef SeedExtensionTrajectoryFilter_H
#define SeedExtensionTrajectoryFilter_H

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class SeedExtensionTrajectoryFilter final : public TrajectoryFilter {
public:
  explicit SeedExtensionTrajectoryFilter() {}

  explicit SeedExtensionTrajectoryFilter(edm::ParameterSet const& pset, edm::ConsumesCollector&)
      : theStrict(pset.getParameter<bool>("strictSeedExtension")),
        thePixel(pset.getParameter<bool>("pixelSeedExtension")),
        theExtension(pset.getParameter<int>("seedExtension")) {}

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    iDesc.add<bool>("strictSeedExtension", false);
    iDesc.add<bool>("pixelSeedExtension", false);
    iDesc.add<int>("seedExtension", 0);
  }

  bool qualityFilter(const Trajectory& traj) const override { return QF(traj); }
  bool qualityFilter(const TempTrajectory& traj) const override { return QF(traj); }

  bool toBeContinued(TempTrajectory& traj) const override { return TBC<TempTrajectory>(traj); }
  bool toBeContinued(Trajectory& traj) const override { return TBC<Trajectory>(traj); }

  std::string name() const override { return "LostHitsFractionTrajectoryFilter"; }

private:
  template <class T>
  bool QF(const T& traj) const {
    return traj.stopReason() != StopReason::SEED_EXTENSION;  // reject tracks killed by seed extension
  }

  template <class T>
  bool TBC(T& traj) const {
    if (theExtension <= 0)
      return true;  // skipping checks explicitly when intended to be disabled is the safest way
    const bool ret = theStrict ? strictTBC(traj) : looseTBC(traj);
    if (!ret)
      traj.setStopReason(StopReason::SEED_EXTENSION);
    return ret;
  }
  template <class T>
  bool looseTBC(const T& traj) const;
  template <class T>
  bool strictTBC(const T& traj) const;

  bool theStrict = false;
  bool thePixel = false;
  int theExtension = 0;
};

template <class T>
bool SeedExtensionTrajectoryFilter::looseTBC(const T& traj) const {
  int nhits = 0;
  if (thePixel) {
    for (const auto& tm : traj.measurements()) {
      if (Trajectory::pixel(*(tm.recHit())))
        ++nhits;
    }
  } else {
    nhits = traj.measurements().size();
  }
  return (nhits > int(traj.seedNHits()) + theExtension) || (0 == traj.lostHits());
}

// strict case as a real seeding: do not allow even inactive
template <class T>
bool SeedExtensionTrajectoryFilter::strictTBC(const T& traj) const {
  const int nhits = thePixel ? traj.foundPixelHits() : traj.foundHits();
  return nhits >= int(traj.seedNHits()) + theExtension;
}

#endif
