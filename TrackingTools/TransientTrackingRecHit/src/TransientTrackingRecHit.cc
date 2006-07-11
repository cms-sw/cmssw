#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"

GlobalPoint TransientTrackingRecHit::globalPosition() const {
  return  (geom_->surface().toGlobal(localPosition()));
}

GlobalError TransientTrackingRecHit::globalPositionError() const {
  return ErrorFrameTransformer().transform( localPositionError(), (geom_->surface()));
}   

edm::OwnVector<const TransientTrackingRecHit> TransientTrackingRecHit::transientHits() const 
{
  // no components by default
  return edm::OwnVector<const TransientTrackingRecHit>();
}
