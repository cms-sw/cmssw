#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiPixelRecHit_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiPixelRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TSiPixelRecHit : public TransientTrackingRecHit{
 public:
  TSiPixelRecHit(const TrackingGeometry * geom, const TrackingRecHit * rh) : TransientTrackingRecHit(geom, rh){}
    
  //
  // fake for the moment
  //
  virtual AlgebraicVector parameters(const TrajectoryStateOnSurface& ts) const {return  hit()->parameters();}
  virtual AlgebraicSymMatrix parametersError(const TrajectoryStateOnSurface& ts) const {  return hit()->parametersError();}
  //
  //
  //
  virtual TransientTrackingRecHit * clone() const {
    return new TSiPixelRecHit(*this);
  }
};



#endif
