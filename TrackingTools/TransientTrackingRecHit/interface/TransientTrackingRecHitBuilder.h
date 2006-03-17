#ifndef TRACKINGTOOLS_TRANSIENTRECHITBUILDER_H
#define TRACKINGTOOLS_TRANSIENTRECHITBUILDER_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TransientTrackingRecHitBuilder {
 public:
   virtual TransientTrackingRecHit * build ( const TrackingRecHit * p)  = 0 ;
};


#endif
