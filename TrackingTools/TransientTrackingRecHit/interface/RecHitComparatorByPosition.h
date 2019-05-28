#ifndef RecHitComparatorByPosition_H
#define RecHitComparatorByPosition_H
#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
class RecHitComparatorByPosition {
public:
  RecHitComparatorByPosition() {}
  bool operator()(const TrackingRecHit* a, const TrackingRecHit* b) const;
  bool equals(const TrackingRecHit* a, const TrackingRecHit* b) const;
};
#endif
