#ifndef RecoTracker_PixelLowPtUtilities_TripletFilter_h
#define RecoTracker_PixelLowPtUtilities_TripletFilter_h

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <vector>

class TrackingRecHit;
class ClusterShapeHitFilter;
class TrackerTopology;
class SiPixelClusterShapeCache;

class TripletFilter {
public:
  explicit TripletFilter(const ClusterShapeHitFilter* iFilter) : theFilter(iFilter) {}
  ~TripletFilter() = default;
  bool checkTrack(const std::vector<const TrackingRecHit*>& recHits,
                  const std::vector<LocalVector>& localDirs,
                  const TrackerTopology* tTopo,
                  const SiPixelClusterShapeCache& clusterShapeCache) const;
  bool checkTrack(const std::vector<const TrackingRecHit*>& recHits,
                  const std::vector<GlobalVector>& globalDirs,
                  const TrackerTopology* tTopo,
                  const SiPixelClusterShapeCache& clusterShapeCache) const;

private:
  const ClusterShapeHitFilter* theFilter;
};

#endif
