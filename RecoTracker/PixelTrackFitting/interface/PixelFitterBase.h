#ifndef RecoTracker_PixelTrackFitting_PixelFitterBase_h
#define RecoTracker_PixelTrackFitting_PixelFitterBase_h

#include "DataFormats/TrackReco/interface/Track.h"

#include <vector>
#include <memory>

class TrackingRegion;
class TrackingRecHit;

class PixelFitterBase {
public:
  virtual ~PixelFitterBase() {}

  virtual std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit*>& hits,
                                           const TrackingRegion& region) const = 0;
};
#endif
