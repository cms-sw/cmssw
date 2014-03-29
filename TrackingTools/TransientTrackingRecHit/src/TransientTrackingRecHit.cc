#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"


#ifdef COUNT_HITS
#include<cstdio>
namespace {

  struct Stat {
    ~Stat() {
      printf("TTRH: %d/%d/%d/%d\n",tot[0],tot[1],tot[2],tot[3]);
    }
    int tot[4]={0};
  };
  Stat stat;
}

void countTTRH(TrackingRecHit::Type type) {
  ++stat.tot[type];
}
#endif





TransientTrackingRecHit::ConstRecHitContainer TransientTrackingRecHit::transientHits() const 
{
  // no components by default
  return ConstRecHitContainer();
}

TransientTrackingRecHit::RecHitPointer 
TransientTrackingRecHit::clone( const TrajectoryStateOnSurface&) const {
  return RecHitPointer(const_cast<TransientTrackingRecHit*>(this));
}
