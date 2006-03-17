#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//
// include all the concrete ones
//
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DMatchedLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"

TkTransientTrackingRecHitBuilder::TkTransientTrackingRecHitBuilder(  const TrackingGeometry* trackingGeometry):
  tGeometry_(trackingGeometry){}

TransientTrackingRecHit* TkTransientTrackingRecHitBuilder::build (const TrackingRecHit * p) {
  if (dynamic_cast<const SiStripRecHit2DLocalPos*>(p)){ 
    return ( new TSiStripRecHit2DLocalPos(tGeometry_, p ) ); 
  } else if (dynamic_cast<const SiStripRecHit2DMatchedLocalPos*>(p)) {
    return ( new TSiStripRecHit2DMatchedLocalPos(tGeometry_, p )); 
  } else if (dynamic_cast<const SiPixelRecHit*>(p)) {
    return ( new TSiPixelRecHit(tGeometry_, p ) ); 
  }
  return 0;
}
