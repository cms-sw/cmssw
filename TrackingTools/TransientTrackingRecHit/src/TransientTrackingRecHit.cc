#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"


const GeomDet * TransientTrackingRecHit::det() const 
{
  return _geom->idToDet(geographicalId());
}
const GeomDetUnit * TransientTrackingRecHit::detUnit() const 
{
  return _geom->idToDetUnit(geographicalId());
}

 GlobalPoint TransientTrackingRecHit::globalPosition() const {
  return  (_geom->idToDet(geographicalId()))->surface().toGlobal(localPosition());
}
 GlobalError TransientTrackingRecHit::globalPositionError() const {
   return ErrorFrameTransformer().transform( localPositionError(), (_geom->idToDet(geographicalId()))->surface());
}   

