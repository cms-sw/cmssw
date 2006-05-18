#ifndef InvalidTransientRecHit_H
#define InvalidTransientRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"

class InvalidTransientRecHit : public GenericTransientTrackingRecHit {
public:

  /// invalid RecHit - has only GeomDet
  explicit InvalidTransientRecHit( const GeomDet* geom);
  
  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

  virtual InvalidTransientRecHit* clone( const TrajectoryStateOnSurface&) const {return clone();}

};

#endif
