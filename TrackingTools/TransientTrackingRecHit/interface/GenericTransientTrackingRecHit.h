#ifndef GenericTransientTrackingRecHit_H
#define GenericTransientTrackingRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class GenericTransientTrackingRecHit: public TransientTrackingRecHit{
 public:
  GenericTransientTrackingRecHit(const GeomDet * geom, const TrackingRecHit * rh) : TransientTrackingRecHit(geom, rh){}
    
  //
  // fake for the moment
  //
  virtual AlgebraicVector parameters(const TrajectoryStateOnSurface& ts) const {return  hit()->parameters();}
  virtual AlgebraicSymMatrix parametersError(const TrajectoryStateOnSurface& ts) const {  return hit()->parametersError();}
  //
  //
  //
  virtual TransientTrackingRecHit * clone() const {
    return new GenericTransientTrackingRecHit(*this);
  }
};

#endif

