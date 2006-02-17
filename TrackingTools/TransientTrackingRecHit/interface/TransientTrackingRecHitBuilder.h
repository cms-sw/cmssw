#ifndef TRANSIENTRECHITBUILDER_H
#define TRANSIENTRECHITBUILDER_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
//#include "TrackingTools/TransientTrackerRecHit2D/interface/TSiStripRecHit2DLocalPos.h"
//#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"

class TransientTrackingRecHitBuilder {

 public:
  
  TransientTrackingRecHit * build (edm::ESHandle<TrackingGeometry> trackingGeometry,  TrackingRecHit * p) {
/*     if (dynamic_cast<SiStripRecHit2DLocalPos*>(p)){ */
/*       return ( new TSiStripRecHit2DLocalPos(trackingGeometry, p ) ); */
/*     } else return 0; //should throw an exception */
    return 0;
  }

};


#endif
