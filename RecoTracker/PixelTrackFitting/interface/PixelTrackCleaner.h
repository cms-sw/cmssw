#ifndef PixelTrackFitting_PixelTrackCleaner_H
#define PixelTrackFitting_PixelTrackCleaner_H

/**
class PixelTrackCleaner:
Discards reconstructed tracks that reflects one real track.
**/

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "RecoTracker/PixelTrackFitting/interface/TracksWithHits.h"
#include <cassert>

class PixelTrackCleaner {
protected:
  explicit PixelTrackCleaner(bool fast = false) : fast_(fast) {}

public:
  using Record = TrajectoryFilter::Record;

  virtual ~PixelTrackCleaner() {}

  bool fast() const { return fast_; }

  // used by HI?
  typedef pixeltrackfitting::TracksWithRecHits TracksWithRecHits;
  virtual TracksWithRecHits cleanTracks(const TracksWithRecHits& tracksWithRecHits) const {
    assert(false);
    return TracksWithRecHits();
  }

  // fast
  using TracksWithTTRHs = pixeltrackfitting::TracksWithTTRHs;
  virtual void cleanTracks(TracksWithTTRHs& tracksWithRecHits) const { assert(false); }

private:
  const bool fast_;
};

#endif
