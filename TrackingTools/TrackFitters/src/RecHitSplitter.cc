#include "TrackingTools/TrackFitters/interface/RecHitSplitter.h"

RecHitSplitter::RecHitContainer
RecHitSplitter::split(const RecHitContainer& hits) const {

  RecHitContainer singles; singles.reserve(2*hits.size());

  for(RecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end();
      ihit++) {
    if(!(**ihit).isValid()) {
      singles.push_back((*ihit));
    } else {
      RecHitContainer shits = (**ihit).transientHits();
      for(RecHitContainer::const_iterator ishit = shits.begin();
	  ishit != shits.end(); ishit++) {
	singles.push_back(*ishit);
      }
    }
  }
  return singles;
}
