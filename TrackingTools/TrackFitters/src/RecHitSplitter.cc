#include "TrackingTools/TrackFitters/interface/RecHitSplitter.h"

edm::OwnVector<TransientTrackingRecHit> RecHitSplitter::split(const edm::OwnVector<TransientTrackingRecHit>& hits) const {

  edm::OwnVector<TransientTrackingRecHit> singles; singles.reserve(2*hits.size());

  for(edm::OwnVector<TransientTrackingRecHit>::const_iterator ihit = hits.begin(); ihit != hits.end();
      ihit++) {
    if(!(*ihit).isValid()) {
      singles.push_back((*ihit).clone());
    } else {
      edm::OwnVector<TransientTrackingRecHit> shits = ihit->clone()->transientHits();
      for(edm::OwnVector<TransientTrackingRecHit>::iterator ishit = shits.begin();
	  ishit != shits.end(); ishit++) {
	singles.push_back(ishit.get());
//       }
      //singles.insert(singles.end(), shits.begin(), shits.end());
      }
    }
  }
  
  return singles;
}
