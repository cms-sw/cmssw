#ifndef _ClusterShapeTrackFilter_h_
#define _ClusterShapeTrackFilter_h_

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilterBase.h"

#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
typedef Vector2DBase<float, GlobalTag> Global2DVector;

//#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <vector>

class TrackerGeometry;
class TrackingRecHit;
class ClusterShapeHitFilter;
class TrackerTopology;
class SiPixelClusterShapeCache;

class ClusterShapeTrackFilter : public PixelTrackFilterBase {
public:
  ClusterShapeTrackFilter(const SiPixelClusterShapeCache *cache,
                          double ptmin,
                          double ptmax,
                          const TrackerGeometry *tracker,
                          const ClusterShapeHitFilter *shape,
                          const TrackerTopology *ttopo);
  ~ClusterShapeTrackFilter() override;
  bool operator()(const reco::Track *, const std::vector<const TrackingRecHit *> &hits) const override;

private:
  float areaParallelogram(const Global2DVector &a, const Global2DVector &b) const;
  std::vector<GlobalVector> getGlobalDirs(const std::vector<GlobalPoint> &globalPoss) const;
  std::vector<GlobalPoint> getGlobalPoss(const std::vector<const TrackingRecHit *> &recHits) const;

  const TrackerGeometry *theTracker;
  const ClusterShapeHitFilter *theFilter;
  const SiPixelClusterShapeCache *theClusterShapeCache;
  const TrackerTopology *tTopo;

  const double ptMin;
  const double ptMax;
};

#endif
