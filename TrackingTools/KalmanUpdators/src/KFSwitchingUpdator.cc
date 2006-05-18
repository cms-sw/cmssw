#include "TrackingTools/KalmanUpdators/interface/KFSwitchingUpdator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

TrajectoryStateOnSurface
KFSwitchingUpdator::update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const {
  if(//aHit.isMatched() || 
     aHit.detUnit()->type().isTrackerPixel()) {
    return localUpdator().update(aTsos, aHit);
  } else {
    return stripUpdator().update(aTsos, aHit);
  }
}
