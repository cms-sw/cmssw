#ifndef HitSplitter_h
#define HitSplitter_h


#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

class HitSplitter {

 public:
  
  void splitHits(edm::OwnVector<TrackingRecHit>::const_iterator beginInput,
		 edm::OwnVector<TrackingRecHit>::const_iterator endInput,
		 const TransientTrackingRecHitBuilder* builder,
		 const TrackingGeometry * theG,
		 bool reverse,
		 TransientTrackingRecHit::RecHitContainer& outputCollection,
		 float& ndof) ;


  void splitHits(trackingRecHit_iterator beginInput,
		 trackingRecHit_iterator endInput,
		 const TransientTrackingRecHitBuilder* builder,
		 const TrackingGeometry * theG,
		 bool reverse,
		 TransientTrackingRecHit::RecHitContainer& outputCollection,
		 float& ndof);

  void doSplit(const TrackingRecHit* hitit,
	       const TransientTrackingRecHitBuilder* builder,
	       const TrackingGeometry * theG,
	       int isInOut,
	       TransientTrackingRecHit::RecHitContainer& outputCollection,
	       float& ndof);


};


#endif
