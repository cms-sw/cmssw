#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"

#include <algorithm>

edm::OwnVector<TransientTrackingRecHit> RecHitSorter::sortHits(const edm::OwnVector<TransientTrackingRecHit>& hits, 
				      const PropagationDirection& dir) const {

  edm::OwnVector<TransientTrackingRecHit> myHits(hits);

  myHits.sort(/*myHits.begin(), myHits.end(),*/ RecHitLessByDet(dir));

  return myHits;

}
