#include "TrackingTools/KalmanUpdators/interface/Chi2SwitchingEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
std::pair<bool,double> 
Chi2SwitchingEstimator::estimate (const TrajectoryStateOnSurface& aTsos,
				  const TransientTrackingRecHit& aHit) const {
  if(//aHit.isMatched() || 
     aHit.detUnit()->type().isTrackerPixel()) {
    return localEstimator().estimate(aTsos, aHit);
  } else {
    return stripEstimator().estimate(aTsos, aHit);
  }    
}
