#ifndef TRACKINGTOOLS_TRANSIENTRECHITBUILDER_H
#define TRACKINGTOOLS_TRANSIENTRECHITBUILDER_H

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"

class TransientTrackingRecHitBuilder {
public:

  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;

  /// build a tracking rechit from an existing rechit
  virtual RecHitPointer build ( const TrackingRecHit * p)  const = 0 ;
  
  /// build a tracking rechit refiting the rechit position and error according to the state estimate
    virtual RecHitPointer build ( const TrackingRecHit * p, const TrajectoryStateOnSurface & state) const { return build(p); } ;
};


#endif
