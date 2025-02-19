#include "TrackingTools/TransientTrackingRecHit/interface/RecHitComparatorByPosition.h"
bool RecHitComparatorByPosition::operator() (const  TrackingRecHit* a, const TrackingRecHit* b) const  {
    float xcut = 0.01;
    float ycut = 0.2;
    if (a->geographicalId()<b->geographicalId()) return true;  
    if (b->geographicalId()<a->geographicalId()) return false;  
    if (a->localPosition().x() < b->localPosition().x() - xcut)  return true;
    if (b->localPosition().x() < a->localPosition().x() - xcut) return false;
    return (a->localPosition().y() < b->localPosition().y() - ycut );
  }
bool RecHitComparatorByPosition::equals(const  TrackingRecHit* a, const TrackingRecHit* b) const  {
    float xcut = 0.01;
    float ycut = 0.2;
    if (a->geographicalId() != b->geographicalId()) return false; 
    if (a->isValid() && b->isValid()) {
        if (fabs(a->localPosition().x() - b->localPosition().x()) >=  xcut)  return false;
        return (fabs(a->localPosition().y() - b->localPosition().y()) <  ycut);
    } else if (!a->isValid() && !b->isValid()) {
        return true;
    } else return false;
}
