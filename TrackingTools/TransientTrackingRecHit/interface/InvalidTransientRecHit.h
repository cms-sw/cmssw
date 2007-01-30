#ifndef InvalidTransientRecHit_H
#define InvalidTransientRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

class InvalidTransientRecHit : public GenericTransientTrackingRecHit {
public:

  typedef TrackingRecHit::Type Type;
  //RC  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

  //RC  virtual InvalidTransientRecHit* clone( const TrajectoryStateOnSurface&) const {return clone();}

  static RecHitPointer build( const GeomDet * geom, Type type=TrackingRecHit::missing) {
    return RecHitPointer( new InvalidTransientRecHit( geom, type ));
  }

private:

  /// invalid RecHit - has only GeomDet and Type
  explicit InvalidTransientRecHit( const GeomDet* geom, Type type);

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

};

#endif
