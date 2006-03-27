#ifndef InvalidTransientRecHit_H
#define InvalidTransientRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class InvalidTransientRecHit : public TransientTrackingRecHit {
public:

  /// invalid RecHit - has only GeomDet
  explicit InvalidTransientRecHit( const GeomDet* geom);
  
  virtual AlgebraicVector parameters(const TrajectoryStateOnSurface& ts) const;

  virtual AlgebraicSymMatrix parametersError(const TrajectoryStateOnSurface& ts) const;

  virtual InvalidTransientRecHit* clone() const {return new InvalidTransientRecHit(*this);}

};

#endif
