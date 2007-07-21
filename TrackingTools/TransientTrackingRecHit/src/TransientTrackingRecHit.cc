#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

GlobalPoint TransientTrackingRecHit::globalPosition() const {
  return  (geom_->surface().toGlobal(localPosition()));
}

GlobalError TransientTrackingRecHit::globalPositionError() const {
  return ErrorFrameTransformer().transform( localPositionError(), (geom_->surface()));
}   

TransientTrackingRecHit::ConstRecHitContainer TransientTrackingRecHit::transientHits() const 
{
  // no components by default
  return ConstRecHitContainer();
}

TransientTrackingRecHit::RecHitPointer 
TransientTrackingRecHit::clone( const TrajectoryStateOnSurface& ts) const {
  return RecHitPointer(const_cast<TransientTrackingRecHit*>(this));
}
