#ifndef RecoTracker_PixelLowPtUtilities_LowPtClusterShapeSeedComparitor_h
#define RecoTracker_PixelLowPtUtilities_LowPtClusterShapeSeedComparitor_h

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

class LowPtClusterShapeSeedComparitor : public SeedComparitor {
public:
  LowPtClusterShapeSeedComparitor(const edm::ParameterSet &ps, edm::ConsumesCollector &iC);
  ~LowPtClusterShapeSeedComparitor() override {}
  void init(const edm::Event &e, const edm::EventSetup &es) override;
  bool compatible(const SeedingHitSet &hits) const override;
  bool compatible(const TrajectoryStateOnSurface &, SeedingHitSet::ConstRecHitPointer hit) const override {
    return true;
  }
  bool compatible(const SeedingHitSet &hits,
                  const GlobalTrajectoryParameters &helixStateAtVertex,
                  const FastHelix &helix) const override {
    return true;
  }

private:
  edm::EDGetTokenT<SiPixelClusterShapeCache> thePixelClusterShapeCacheToken;
  edm::Handle<SiPixelClusterShapeCache> thePixelClusterShapeCache;
  std::string theShapeFilterLabel_;
  const edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> clusterShapeHitFilterESToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyESToken_;
  const ClusterShapeHitFilter *clusterShapeHitFilter_ = nullptr;
  const TrackerTopology *trackerTopology_ = nullptr;
};

#endif
