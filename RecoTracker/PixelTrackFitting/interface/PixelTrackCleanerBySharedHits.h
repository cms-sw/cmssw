#ifndef RecoTracker_PixelTrackFitting_PixelTrackCleanerBySharedHits_h
#define RecoTracker_PixelTrackFitting_PixelTrackCleanerBySharedHits_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoTracker/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include <utility>
#include <vector>

class TrackerTopology;

class PixelTrackCleanerBySharedHits final : public PixelTrackCleaner {
public:
  PixelTrackCleanerBySharedHits(bool useQuadrupletAlgo);

  ~PixelTrackCleanerBySharedHits() override;

  using TrackWithTTRHs = pixeltrackfitting::TrackWithTTRHs;
  using TracksWithTTRHs = pixeltrackfitting::TracksWithTTRHs;
  void cleanTracks(TracksWithTTRHs& tracksWithRecHits) const override;

private:
  const bool useQuadrupletAlgo_;
};

#endif
