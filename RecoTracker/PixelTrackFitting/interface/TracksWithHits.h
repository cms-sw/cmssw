#ifndef RecoTracker_PixelTrackFitting_TracksWithHits_h
#define RecoTracker_PixelTrackFitting_TracksWithHits_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

#include <vector>

namespace pixeltrackfitting {
  typedef std::pair<reco::Track *, std::vector<const TrackingRecHit *> > TrackWithRecHits;
  typedef std::pair<reco::Track *, SeedingHitSet> TrackWithTTRHs;
  typedef std::vector<TrackWithRecHits> TracksWithRecHits;
  typedef std::vector<TrackWithTTRHs> TracksWithTTRHs;
}  // namespace pixeltrackfitting

#endif
