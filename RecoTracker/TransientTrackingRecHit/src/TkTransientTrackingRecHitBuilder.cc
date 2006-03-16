#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//
// include all the concrete ones
//
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DMatchedLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"

TkTransientTrackingRecHitBuilder::TkTransientTrackingRecHitBuilder(edm::ESHandle<TrackingGeometry> trackingGeometry):
  tGeometry_(trackingGeometry){}

TransientTrackingRecHit* TkTransientTrackingRecHitBuilder::build (TrackingRecHit * p) {
  if (dynamic_cast<SiStripRecHit2DLocalPos*>(p)){ 
    return ( new TSiStripRecHit2DLocalPos(tGeometry_, p ) ); 
  } else if (dynamic_cast<SiStripRecHit2DMatchedLocalPos*>(p)) {
    return ( new TSiStripRecHit2DMatchedLocalPos(tGeometry_, p )); 
  } else if (dynamic_cast<SiPixelRecHit*>(p)) {
    return ( new TSiPixelRecHit(tGeometry_, p ) ); 
  }
}
