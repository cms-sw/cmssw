#ifndef InvalidTransientRecHit_H
#define InvalidTransientRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

class InvalidTransientRecHit : public GenericTransientTrackingRecHit {
public:

  typedef TrackingRecHit::Type Type;
  //RC  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

  //RC  virtual InvalidTransientRecHit* clone( const TrajectoryStateOnSurface&) const {return clone();}

  static RecHitPointer build( const GeomDet * geom, Type type=TrackingRecHit::missing, const DetLayer * layer=0) {
    return RecHitPointer( new InvalidTransientRecHit( geom, layer, type ));
  }

  const Surface* surface() const ;/* { */
/*     const BoundSurface* ret = 0; */
/*     if (det() != 0 ) { LogTrace("TrackFitters") <<"A "<<&(det()->surface()); return &(det()->surface()); } */
/*     else if (layer_ != 0) { LogTrace("TrackFitters") <<"B"; return &(layer_->surface()); } */
/*     else { LogTrace("TrackFitters") <<"C"; return ret; } */
/*     //{throw cms::Exception("InvalidTransientRecHit") << "Trying to access surface of an Invalid hit without GeomDet or DetLayer";} */
/*   } */

private:
  const DetLayer * layer_;
  /// invalid RecHit - has only GeomDet and Type
  explicit InvalidTransientRecHit( const GeomDet* geom, const DetLayer * layer, Type type);

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

};

#endif
