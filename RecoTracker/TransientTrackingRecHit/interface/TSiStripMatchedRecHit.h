#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"

class TSiStripMatchedRecHit : public GenericTransientTrackingRecHit{
public:

//    TSiStripMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh) : 
//      GenericTransientTrackingRecHit(geom, rh){}

//   virtual TSiStripMatchedRecHit* clone() const {
//     return new TSiStripMatchedRecHit(*this);
//   }

//   virtual TSiStripMatchedRecHit* clone (const TrajectoryStateOnSurface& ts) const {
//     return clone();
//   }

  const GeomDetUnit* detUnit() const {return 0;}

  static RecHitPointer build( const GeomDet * geom, const TrackingRecHit * rh) {
    return RecHitPointer( new TSiStripMatchedRecHit( geom, rh));
  }

private:

  TSiStripMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh) : 
     GenericTransientTrackingRecHit(geom, *rh){}

  virtual TSiStripMatchedRecHit* clone() const {
    return new TSiStripMatchedRecHit(*this);
  }

};



#endif
