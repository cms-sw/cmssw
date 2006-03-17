#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit2DMatchedLocalPos_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit2DMatchedLocalPos_H


#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"


class TSiStripRecHit2DMatchedLocalPos : public TransientTrackingRecHit{
 public:
  TSiStripRecHit2DMatchedLocalPos(const TrackingGeometry * geom, const TrackingRecHit * rh) : TransientTrackingRecHit(geom, rh){}

  //
  // fake for the moment
  //
  virtual AlgebraicVector parameters(const TrajectoryStateOnSurface& ts) const {return  hit()->parameters();}
  virtual AlgebraicSymMatrix parametersError(const TrajectoryStateOnSurface& ts) const {  return hit()->parametersError();}
  //
  //
  //
  virtual TransientTrackingRecHit * clone() const {
    return new TSiStripRecHit2DMatchedLocalPos(*this);
  }
};



#endif
