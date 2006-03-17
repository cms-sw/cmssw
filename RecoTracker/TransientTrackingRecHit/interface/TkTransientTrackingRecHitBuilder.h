#ifndef RECOTRACKER_TRANSIENTRECHITBUILDER_H
#define RECOTRACKER_TRANSIENTRECHITBUILDER_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

class TkTransientTrackingRecHitBuilder : public TransientTrackingRecHitBuilder {

 public:
  TkTransientTrackingRecHitBuilder( const TrackingGeometry* trackingGeometry);
  TransientTrackingRecHit * build (const TrackingRecHit * p) ;
 private:
  const TrackingGeometry* tGeometry_;
};


#endif
