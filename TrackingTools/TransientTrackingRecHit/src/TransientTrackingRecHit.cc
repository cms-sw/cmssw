#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

const GeomDetUnit * GenericTransientTrackingRecHit::detUnit() const
{
  return dynamic_cast<const GeomDetUnit*>(det());
}


GlobalPoint TransientTrackingRecHit::globalPosition() const {
  return  (surface()->toGlobal(localPosition()));
}

GlobalError TransientTrackingRecHit::globalPositionError() const {
  return ErrorFrameTransformer().transform( localPositionError(), *surface() );
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
