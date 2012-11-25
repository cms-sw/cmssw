#ifndef InvalidTransientRecHit_H
#define InvalidTransientRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitByValue.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

class InvalidTransientRecHit GCC11_FINAL : public TransientTrackingRecHitByValue<InvalidTrackingRecHit> {
public:
  typedef TransientTrackingRecHitByValue<InvalidTrackingRecHit> Base;
  typedef TrackingRecHit::Type Type;

  static RecHitPointer build( const GeomDet * geom, Type type=TrackingRecHit::missing, const DetLayer * layer=nullptr) {
    return RecHitPointer( new InvalidTransientRecHit( geom, layer, type ));
  }

  const Surface* surface() const {  return  surface_; }

private:
   Surface const * surface_;
  /// invalid RecHit - has only GeomDet and Type
  InvalidTransientRecHit( const GeomDet* geom, const DetLayer * layer, Type type) :
    Base( geom, InvalidTrackingRecHit( geom == nullptr ? DetId(0) : geom->geographicalId(), type)), 
    surface_(geom ? &(det()->surface()) : ( layer ?  &(layer->surface()) : nullptr))
    {}

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

};

#endif
