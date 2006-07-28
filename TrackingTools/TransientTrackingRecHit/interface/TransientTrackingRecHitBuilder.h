#ifndef TRACKINGTOOLS_TRANSIENTRECHITBUILDER_H
#define TRACKINGTOOLS_TRANSIENTRECHITBUILDER_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TransientTrackingRecHitBuilder {
public:

  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;

  virtual RecHitPointer build ( const TrackingRecHit * p)  const = 0 ;
};


#endif
