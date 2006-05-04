#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHitBuilder.h"
//
// include the concrete GenericTransientTrackingRecHit
//
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"


GenericTransientTrackingRecHitBuilder::GenericTransientTrackingRecHitBuilder(  const TrackingGeometry* trackingGeometry):
  theTrackingGeometry(trackingGeometry){}

TransientTrackingRecHit* GenericTransientTrackingRecHitBuilder::build (const TrackingRecHit * p) {
  if (dynamic_cast<const GenericTransientTrackingRecHit*>(p)) {
    return ( new GenericTransientTrackingRecHit(theTrackingGeometry->idToDet(p->geographicalId()), p ) ); 
  }else if (dynamic_cast<const InvalidTrackingRecHit*>(p)){
    return ( new InvalidTransientRecHit((p->geographicalId().rawId() == 0 ? 0 : 
					theTrackingGeometry->idToDet(p->geographicalId())) ));
  }
  return 0;
}


