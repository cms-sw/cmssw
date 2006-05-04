#ifndef TRACKINGTOOLS_GENERICTRANSIENTTRACKINGRECHITBUILDER_H
#define TRACKINGTOOLS_GENERICTRANSIENTTRACKINGRECHITBUILDER_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

class GenericTransientTrackingRecHitBuilder : public TransientTrackingRecHitBuilder {

 public:
  GenericTransientTrackingRecHitBuilder( const TrackingGeometry* trackingGeometry);
  TransientTrackingRecHit * build (const TrackingRecHit * p) ;
 private:
  const TrackingGeometry* theTrackingGeometry;
};


#endif

