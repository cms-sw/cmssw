#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"

#include <algorithm>

RecHitSorter::RecHitContainer RecHitSorter::sortHits(const RecHitContainer& hits, 
						     const PropagationDirection& dir) const {

  RecHitContainer myHits(hits);

  myHits.sort(/*myHits.begin(), myHits.end(),*/ RecHitLessByDet(dir));

  return myHits;

}
