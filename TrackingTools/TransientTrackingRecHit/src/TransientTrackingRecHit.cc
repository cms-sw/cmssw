#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

TransientTrackingRecHit::TransientTrackingRecHit( const GeomDet* geom) {
  _geom = geom ;
  _trackingRecHit = new InvalidTrackingRecHit(geom->geographicalId());
}

const GeomDet * TransientTrackingRecHit::det() const 
{
  return _geom;
}
const GeomDetUnit * TransientTrackingRecHit::detUnit() const 
{
  return dynamic_cast<const GeomDetUnit*>(_geom);
}

 GlobalPoint TransientTrackingRecHit::globalPosition() const {
   return  (_geom->surface().toGlobal(localPosition()));
}
 GlobalError TransientTrackingRecHit::globalPositionError() const {
   return ErrorFrameTransformer().transform( localPositionError(), (_geom->surface()));
}   

