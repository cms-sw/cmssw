#include "TrackingTools/KalmanUpdators/interface/KFSwitching1DUpdator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

TrajectoryStateOnSurface
KFSwitching1DUpdator::update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const {
  if(//aHit.isMatched() || 
     aHit.detUnit()->type().isTrackerPixel() || aHit.detUnit()->type().isEndcap()) {
    return localUpdator().update(aTsos, aHit);
  } else {
    return stripUpdator().update(aTsos, aHit);
  }
}
