#ifndef RecoPixelVertexing_PixelTrackFitting_PixelTrackCleanerWrapper_H
#define RecoPixelVertexing_PixelTrackFitting_PixelTrackCleanerWrapper_H

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoTracker/PixelTrackFitting/interface/TracksWithHits.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <map>
#include <cassert>

class PixelTrackCleanerWrapper {
public:
  PixelTrackCleanerWrapper(const PixelTrackCleaner *tc) : theCleaner(tc) {}
  pixeltrackfitting::TracksWithTTRHs clean(const pixeltrackfitting::TracksWithTTRHs &initialT_TTRHs) const {
    pixeltrackfitting::TracksWithRecHits initialT_TRHs;
    std::map<const TrackingRecHit *, SeedingHitSet::ConstRecHitPointer> hitMap;

    for (pixeltrackfitting::TracksWithTTRHs::const_iterator it = initialT_TTRHs.begin(), iend = initialT_TTRHs.end();
         it < iend;
         ++it) {
      SeedingHitSet ttrhs = it->second;
      std::vector<const TrackingRecHit *> trhs;
      for (unsigned int i = 0, n = ttrhs.size(); i < n; ++i) {
        const TrackingRecHit *trh = ttrhs[i]->hit();
        trhs.push_back(trh);
        hitMap[trh] = ttrhs[i];
      }
      initialT_TRHs.push_back(pixeltrackfitting::TrackWithRecHits(it->first, trhs));
    }

    pixeltrackfitting::TracksWithRecHits finalT_TRHs = theCleaner->cleanTracks(initialT_TRHs);
    pixeltrackfitting::TracksWithTTRHs finalT_TTRHs;

    for (pixeltrackfitting::TracksWithRecHits::const_iterator it = finalT_TRHs.begin(), iend = finalT_TRHs.end();
         it < iend;
         ++it) {
      const std::vector<const TrackingRecHit *> &trhs = it->second;
      assert(!(trhs.size() < 2));
      if (trhs.size() < 2)
        continue;
      SeedingHitSet ttrhs(hitMap[trhs[0]],
                          hitMap[trhs[1]],
                          trhs.size() > 2 ? hitMap[trhs[2]] : SeedingHitSet::nullPtr(),
                          trhs.size() > 3 ? hitMap[trhs[3]] : SeedingHitSet::nullPtr());

      finalT_TTRHs.push_back(pixeltrackfitting::TrackWithTTRHs(it->first, ttrhs));
    }
    return finalT_TTRHs;
  }

private:
  const PixelTrackCleaner *theCleaner;
};
#endif
