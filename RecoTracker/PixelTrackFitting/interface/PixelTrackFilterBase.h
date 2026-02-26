#ifndef RecoTracker_PixelTrackFitting_PixelTrackFilterBase_h
#define RecoTracker_PixelTrackFitting_PixelTrackFilterBase_h

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace edm {
  class Event;
  class EventSetup;
  class ConsumesCollector;
}  // namespace edm

#include <vector>

class PixelTrackFilterBase {
public:
  virtual ~PixelTrackFilterBase() {}
  typedef std::vector<const TrackingRecHit *> Hits;
  virtual bool operator()(const reco::Track *, const Hits &) const = 0;
};
#endif
