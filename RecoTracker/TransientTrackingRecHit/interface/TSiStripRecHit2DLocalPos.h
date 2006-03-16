#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit2DLocalPos_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit2DLocalPos_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"

class TSiStripRecHit2DLocalPos : public TransientTrackingRecHit{
 public:
   TSiStripRecHit2DLocalPos (edm::ESHandle<TrackingGeometry> geom, TrackingRecHit * rh) : TransientTrackingRecHit(geom, rh){}

  //
  // fake for the moment
  //
  virtual AlgebraicVector parameters(const TrajectoryStateOnSurface& ts) const {return  hit()->parameters();}
  virtual AlgebraicSymMatrix parametersError(const TrajectoryStateOnSurface& ts) const {  return hit()->parametersError();}
  //
  //
  //
  virtual TransientTrackingRecHit * clone() const {
    return new TSiStripRecHit2DLocalPos(*this);
  }
};



#endif
