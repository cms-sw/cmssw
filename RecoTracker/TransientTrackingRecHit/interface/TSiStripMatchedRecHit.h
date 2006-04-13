#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TSiStripMatchedRecHit : public TransientTrackingRecHit{
 public:
   TSiStripMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh) : 
     TransientTrackingRecHit(geom, rh){}

  //
  // fake for the moment
  //
  virtual AlgebraicVector parameters(const TrajectoryStateOnSurface& ts) const {return  hit()->parameters();}
  virtual AlgebraicSymMatrix parametersError(const TrajectoryStateOnSurface& ts) const {  return hit()->parametersError();}
  //
  //
  //
  virtual TransientTrackingRecHit * clone() const {
    return new TSiStripMatchedRecHit(*this);
  }
};



#endif
