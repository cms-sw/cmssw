#include "RecoTracker/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include <cassert>
#include <cstdint>

using namespace std;
using namespace reco;
using namespace pixeltrackfitting;

PixelTrackCleanerBySharedHits::PixelTrackCleanerBySharedHits(bool useQuadrupletAlgo)
    : PixelTrackCleaner(true),  // to mark this as fast algo
      useQuadrupletAlgo_(useQuadrupletAlgo) {}

PixelTrackCleanerBySharedHits::~PixelTrackCleanerBySharedHits() {}

void PixelTrackCleanerBySharedHits::cleanTracks(TracksWithTTRHs& trackHitPairs) const {
  LogDebug("PixelTrackCleanerBySharedHits") << "Cleanering tracks"
                                            << "\n";
  unsigned int size = trackHitPairs.size();
  if (size <= 1)
    return;

  // sort (stabilize cleaning)
  uint16_t score[size];
  unsigned int ind[size];
  for (auto i = 0U; i < size; ++i) {
    ind[i] = i;
    score[i] = 32000 - std::min(32000, int(trackHitPairs[i].first->chi2() * 100.f));  // chi2: smaller is better
    if (trackHitPairs[i].second.size() == 4)
      score[i] += 32001;  //  s4 always better than s3
  }
  std::sort(ind, ind + size, [&](unsigned int i, unsigned int j) { return score[i] > score[j]; });

  auto kill = [&](unsigned int k) {
    delete trackHitPairs[k].first;
    trackHitPairs[k].first = nullptr;
  };

  // sorted: first is always better!

  // first loop: only first    two hits....
  for (auto i = 0U; i < size; ++i) {
    auto iTrack1 = ind[i];
    auto track1 = trackHitPairs[iTrack1].first;
    if (!track1)
      continue;
    auto const& recHits1 = trackHitPairs[iTrack1].second;
    for (auto j = i + 1; j < size; ++j) {
      auto iTrack2 = ind[j];
      auto track2 = trackHitPairs[iTrack2].first;
      if (!track2)
        continue;
      auto const& recHits2 = trackHitPairs[iTrack2].second;
      if (recHits1[0] != recHits2[0])
        continue;
      if (recHits1[1] != recHits2[1])
        continue;
      kill(iTrack2);
    }  // tk2
  }    // tk1

  // second loop: first and third hits....
  for (auto i = 0U; i < size; ++i) {
    auto iTrack1 = ind[i];
    auto track1 = trackHitPairs[iTrack1].first;
    if (!track1)
      continue;
    auto const& recHits1 = trackHitPairs[iTrack1].second;
    if (recHits1.size() < 3)
      continue;
    for (auto j = i + 1; j < size; ++j) {
      auto iTrack2 = ind[j];
      auto track2 = trackHitPairs[iTrack2].first;
      if (!track2)
        continue;
      auto const& recHits2 = trackHitPairs[iTrack2].second;
      if (recHits2.size() < 3)
        continue;
      if (recHits1[0] != recHits2[0])
        continue;
      if (recHits1[2] != recHits2[2])
        continue;
      kill(iTrack2);
    }  // tk2
  }    // tk1

  // final loop: all the rest
  for (auto i = 0U; i < size; ++i) {
    auto iTrack1 = ind[i];
    auto track1 = trackHitPairs[iTrack1].first;
    if (!track1)
      continue;
    auto const& recHits1 = trackHitPairs[iTrack1].second;
    auto s1 = recHits1.size();
    for (auto j = i + 1; j < size; ++j) {
      auto iTrack2 = ind[j];
      auto track2 = trackHitPairs[iTrack2].first;
      if (!track2)
        continue;
      auto const& recHits2 = trackHitPairs[iTrack2].second;
      auto s2 = recHits2.size();
      auto commonRecHits = 0U;
      auto f2 = 0U;
      for (auto iRecHit1 = 0U; iRecHit1 < s1; ++iRecHit1) {
        for (auto iRecHit2 = f2; iRecHit2 < s2; ++iRecHit2) {
          if (recHits1[iRecHit1] == recHits2[iRecHit2]) {
            ++commonRecHits;
            f2 = iRecHit2 + 1;
            break;
          }  // if a hit is common, no other can be the same!
        }
        if (commonRecHits > 1)
          break;
      }
      if (useQuadrupletAlgo_) {
        if (commonRecHits >= 1) {
          if (s1 == 3 || commonRecHits > 1) {
            kill(iTrack2);
          }
        }
      } else if (commonRecHits > 1) {
        kill(iTrack2);
      }
    }  // tk2
  }    //tk1

  trackHitPairs.erase(
      std::remove_if(trackHitPairs.begin(), trackHitPairs.end(), [&](TrackWithTTRHs& v) { return nullptr == v.first; }),
      trackHitPairs.end());
}
