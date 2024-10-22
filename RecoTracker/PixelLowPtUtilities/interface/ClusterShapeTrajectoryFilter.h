#ifndef RecoTracker_PixelLowPtUtilities_ClusterShapeTrajectoryFilter_h
#define RecoTracker_PixelLowPtUtilities_ClusterShapeTrajectoryFilter_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
}  // namespace edm

class SiPixelRecHit;
class SiStripRecHit2D;
class GlobalTrackingGeometry;
class MagneticField;
class SiPixelLorentzAngle;
class SiStripLorentzAngle;
class ClusterShapeHitFilter;
class SiPixelClusterShapeCache;

class ClusterShapeTrajectoryFilter : public TrajectoryFilter {
public:
  ClusterShapeTrajectoryFilter(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);
  ~ClusterShapeTrajectoryFilter() override;

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  void setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  bool qualityFilter(const TempTrajectory&) const override;
  bool qualityFilter(const Trajectory&) const override;

  bool toBeContinued(TempTrajectory&) const override;
  bool toBeContinued(Trajectory&) const override;

  std::string name() const override { return "ClusterShapeTrajectoryFilter"; }

private:
  edm::EDGetTokenT<SiPixelClusterShapeCache> theCacheToken;
  edm::ESGetToken<ClusterShapeHitFilter, TrajectoryFilter::Record> theFilterToken;
  const SiPixelClusterShapeCache* theCache;
  const ClusterShapeHitFilter* theFilter;
};

#endif
