#ifndef InvalidTransientRecHit_H
#define InvalidTransientRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"

class InvalidTransientRecHit : public GenericTransientTrackingRecHit {
public:

  //RC  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

  //RC  virtual InvalidTransientRecHit* clone( const TrajectoryStateOnSurface&) const {return clone();}

  static RecHitPointer build( const GeomDet * geom) {
    return RecHitPointer( new InvalidTransientRecHit( geom));
  }

private:

  /// invalid RecHit - has only GeomDet
  explicit InvalidTransientRecHit( const GeomDet* geom);

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

};

#endif
