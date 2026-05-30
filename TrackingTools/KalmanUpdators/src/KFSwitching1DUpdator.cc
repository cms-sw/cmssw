#include "TrackingTools/KalmanUpdators/interface/KFSwitching1DUpdator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonTopologies/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"

TrajectoryStateOnSurface KFSwitching1DUpdator::update(const TSOS& aTsos, const TrackingRecHit& aHit) const {
  if (!aHit.detUnit() || aHit.detUnit()->type().isTrackerPixel() ||
      (!theDoEndCap && aHit.detUnit()->type().isEndcap())) {
    return localUpdator().update(aTsos, aHit);
  } else {
    return stripUpdator().update(aTsos, aHit);
  }
}
