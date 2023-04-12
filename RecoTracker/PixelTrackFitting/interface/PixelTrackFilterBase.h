#ifndef RecoPixelVertexing_PixelTrackFitting_PixelTrackFilterBase_H
#define RecoPixelVertexing_PixelTrackFitting_PixelTrackFilterBase_H

namespace reco {
  class Track;
}
namespace edm {
  class Event;
  class EventSetup;
  class ConsumesCollector;
}  // namespace edm
class TrackingRecHit;

#include <vector>

class PixelTrackFilterBase {
public:
  virtual ~PixelTrackFilterBase() {}
  typedef std::vector<const TrackingRecHit *> Hits;
  virtual bool operator()(const reco::Track *, const Hits &) const = 0;
};
#endif
