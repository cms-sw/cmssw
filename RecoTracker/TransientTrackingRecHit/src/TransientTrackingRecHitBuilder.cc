#include "RecoTracker/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
//
// include all the concrete ones
//
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DMatchedLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"

TransientTrackingRecHitBuilder::TransientTrackingRecHitBuilder(const TrackingGeometry* trackingGeometry):
tGeometry_;(trackingGeometry) {}

TransientTrackingRecHit* TransientTrackingRecHitBuilder::build (TrackingRecHit * p) {
  if (dynamic_cast<SiStripRecHit2DLocalPos*>(p)){ 
    return ( new TSiStripRecHit2DLocalPos(tGeometry_, p ) ); 
  } else if (dynamic_cast<SiStripRecHit2DMatchedLocalPos*>(p)) {
    return ( new TSiStripRecHit2DMatchedLocalPos(tGeometry_, p )); 
  } else if (dynamic_cast<SiPixelRecHit*>(p)) {
    return ( new TSiPixelRecHit(tGeometry_, p ) ); 
  }
}
