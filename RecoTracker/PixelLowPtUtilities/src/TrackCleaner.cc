#include "RecoTracker/PixelLowPtUtilities/interface/TrackCleaner.h"
#include "RecoTracker/PixelLowPtUtilities/interface/HitInfo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

using namespace std;
using namespace pixeltrackfitting;

/*****************************************************************************/
class HitComparatorByRadius {  // No access to geometry!
public:
  HitComparatorByRadius(const TrackerTopology *tTopo) { tTopo_ = tTopo; }

private:
  const TrackerTopology *tTopo_;

public:
  bool operator()(const TrackingRecHit *a, const TrackingRecHit *b) const {
    DetId i1 = a->geographicalId();
    DetId i2 = b->geographicalId();

    bool isBarrel = (i1.subdetId() == int(PixelSubdetector::PixelBarrel));

    if (i1.subdetId() != i2.subdetId()) {
      return isBarrel;
    } else {
      if (isBarrel) {
        int r1 = (tTopo_->pxbLayer(i1) - 1) * 2 + (tTopo_->pxbLadder(i1) - 1) % 2;
        int r2 = (tTopo_->pxbLayer(i2) - 1) * 2 + (tTopo_->pxbLadder(i2) - 1) % 2;

        return (r1 < r2);
      } else {
        int r1 = (tTopo_->pxfDisk(i1) - 1) * 2 + (tTopo_->pxfPanel(i1) - 1) % 2;
        int r2 = (tTopo_->pxfDisk(i2) - 1) * 2 + (tTopo_->pxfPanel(i2) - 1) % 2;

        return (r1 < r2);
      }
    }
  }
};

/*****************************************************************************/
class HitComparator {
public:
  bool operator()(const TrackingRecHit *a, const TrackingRecHit *b) const {
    if (a->geographicalId() < b->geographicalId())
      return true;
    if (b->geographicalId() < a->geographicalId())
      return false;

    if (a->localPosition().x() < b->localPosition().x() - 1e-5)
      return true;
    if (b->localPosition().x() < a->localPosition().x() - 1e-5)
      return false;

    if (a->localPosition().y() < b->localPosition().y() - 1e-5)
      return true;
    if (b->localPosition().y() < a->localPosition().y() - 1e-5)
      return false;

    return false;
  }
};

/*****************************************************************************/
TrackCleaner::TrackCleaner(const TrackerTopology *tTopo) : tTopo_(tTopo) {}

/*****************************************************************************/
TrackCleaner::~TrackCleaner() {}

/*****************************************************************************/
bool TrackCleaner::areSame(const TrackingRecHit *a, const TrackingRecHit *b) const {
  if (a->geographicalId() != b->geographicalId())
    return false;

  if (fabs(a->localPosition().x() - b->localPosition().x()) < 1e-5 &&
      fabs(a->localPosition().y() - b->localPosition().y()) < 1e-5)
    return true;
  else
    return false;
}

/*****************************************************************************/
bool TrackCleaner::isCompatible(const DetId &i1, const DetId &i2) const {
  // different subdet
  if (i1.subdetId() != i2.subdetId())
    return true;

  if (i1.subdetId() == int(PixelSubdetector::PixelBarrel)) {  // barrel

    if (tTopo_->pxbLayer(i1) != tTopo_->pxbLayer(i2))
      return true;

    int dphi = abs(int(tTopo_->pxbLadder(i1) - tTopo_->pxbLadder(i2)));
    //FIXME: non-phase-0 wrap-around is wrong (sh/be 12, 28, 44, 64 in phase-1)
    // the harm is somewhat minimal with some excess of duplicates.
    constexpr int max[4] = {20, 32, 44, 64};
    auto aLayer = tTopo_->pxbLayer(i1) - 1;
    assert(aLayer < 4);
    if (dphi > max[aLayer] / 2)
      dphi = max[aLayer] - dphi;

    int dz = abs(int(tTopo_->pxbModule(i1) - tTopo_->pxbModule(i2)));

    if (dphi == 1 && dz <= 1)
      return true;
  } else {  // endcap

    if (tTopo_->pxfSide(i1) != tTopo_->pxfSide(i2) || tTopo_->pxfDisk(i1) != tTopo_->pxfDisk(i2))
      return true;

    int dphi = abs(int(tTopo_->pxfBlade(i1) - tTopo_->pxfBlade(i2)));
    constexpr int max = 24;  //FIXME: non-phase-0 wrap-around is wrong
    if (dphi > max / 2)
      dphi = max - dphi;

    int dr = abs(int(((tTopo_->pxfModule(i1) - 1) * 2 + (tTopo_->pxfPanel(i1) - 1)) -
                     ((tTopo_->pxfModule(i2) - 1) * 2 + (tTopo_->pxfPanel(i2) - 1))));

    if (dphi <= 1 && dr <= 1 && !(dphi == 0 && dr == 0))
      return true;
  }

  return false;
}

/*****************************************************************************/
bool TrackCleaner::canBeMerged(const vector<const TrackingRecHit *> &recHitsA,
                               const vector<const TrackingRecHit *> &recHitsB) const {
  bool ok = true;

  for (vector<const TrackingRecHit *>::const_iterator recHitA = recHitsA.begin(); recHitA != recHitsA.end(); recHitA++)
    for (vector<const TrackingRecHit *>::const_iterator recHitB = recHitsB.begin(); recHitB != recHitsB.end();
         recHitB++)
      if (!areSame(*recHitA, *recHitB))
        if (!isCompatible((*recHitA)->geographicalId(), (*recHitB)->geographicalId()))
          ok = false;

  return ok;
}

/*****************************************************************************/
TracksWithRecHits TrackCleaner::cleanTracks(const TracksWithRecHits &tracks_) const {
  // Local copy
  TracksWithRecHits tracks = tracks_;

  typedef map<const TrackingRecHit *, vector<unsigned int>, HitComparator> RecHitMap;

  vector<bool> keep(tracks.size(), true);

  int changes;

  LogTrace("MinBiasTracking") << " [TrackCleaner] initial tracks : " << tracks.size();

  for (unsigned int i = 0; i < tracks.size(); i++)
    LogTrace("TrackCleaner") << "   Track #" << i << " : " << HitInfo::getInfo(tracks[i].second, tTopo_);

  do {
    changes = 0;

    RecHitMap recHitMap;

    /*
  LogTrace("MinBiasTracking")
    << " [TrackCleaner] fill rechit map";
*/

    // Fill the rechit map
    for (unsigned int i = 0; i < tracks.size(); i++)
      if (keep[i]) {
        for (vector<const TrackingRecHit *>::const_iterator recHit = tracks[i].second.begin();
             recHit != tracks[i].second.end();
             recHit++)
          recHitMap[*recHit].push_back(i);
      }

    // Look at each track
    typedef map<unsigned int, int, less<unsigned int> > TrackMap;

    for (unsigned int i = 0; i < tracks.size(); i++) {
      // Skip if 'i' already removed
      if (!keep[i])
        continue;

      /*

    bool addedNewHit = false;
    do
    {
*/
      TrackMap trackMap;

      // Go trough all rechits of this track
      for (vector<const TrackingRecHit *>::const_iterator recHit = tracks[i].second.begin();
           recHit != tracks[i].second.end();
           recHit++) {
        // Get tracks sharing this rechit
        vector<unsigned int> sharing = recHitMap[*recHit];

        for (vector<unsigned int>::iterator j = sharing.begin(); j != sharing.end(); j++)
          if (i < *j)
            trackMap[*j]++;
      }

      // Check for tracks with shared rechits
      for (TrackMap::iterator sharing = trackMap.begin(); sharing != trackMap.end(); sharing++) {
        unsigned int j = (*sharing).first;
        if (!keep[i] || !keep[j])
          continue;

        if (tracks[i].second.size() >= 3) {  // triplet tracks
          if ((*sharing).second > min(int(tracks[i].second.size()),
                                      int(tracks[j].second.size())) /
                                      2) {                          // more than min(hits1,hits2)/2 rechits are shared
            if (canBeMerged(tracks[i].second, tracks[j].second)) {  // no common layer
              // merge tracks, add separate hits of the second to the first one
              for (vector<const TrackingRecHit *>::const_iterator recHit = tracks[j].second.begin();
                   recHit != tracks[j].second.end();
                   recHit++) {
                bool ok = true;
                for (vector<const TrackingRecHit *>::const_iterator recHitA = tracks[i].second.begin();
                     recHitA != tracks[i].second.end();
                     recHitA++)
                  if (areSame(*recHit, *recHitA))
                    ok = false;

                if (ok) {
                  tracks[i].second.push_back(*recHit);
                  recHitMap[*recHit].push_back(i);

                  sort(tracks[i].second.begin(), tracks[i].second.end(), HitComparatorByRadius(tTopo_));

                  //addedNewHit = true;
                }
              }

              LogTrace("TrackCleaner") << "   Merge #" << i << " #" << j << ", first now has "
                                       << tracks[i].second.size();

              // Remove second track
              keep[j] = false;

              changes++;
            } else {  // there is a common layer, keep smaller impact
              if (fabs(tracks[i].first->d0()) < fabs(tracks[j].first->d0()))
                keep[j] = false;
              else
                keep[i] = false;

              LogTrace("TrackCleaner") << "   Clash #" << i << " #" << j << " keep lower d0 " << tracks[i].first->d0()
                                       << " " << tracks[j].first->d0() << ", keep #"
                                       << (keep[i] ? i : (keep[j] ? j : 9999));

              changes++;
            }
          } else {  // note more than 50%, but at least two are shared
            if ((*sharing).second > 1) {
              if (tracks[i].second.size() != tracks[j].second.size()) {  // keep longer
                if (tracks[i].second.size() > tracks[j].second.size())
                  keep[j] = false;
                else
                  keep[i] = false;
                changes++;

                LogTrace("TrackCleaner") << "   Sharing " << (*sharing).second << " remove by size";
              } else {  // keep smaller impact
                if (fabs(tracks[i].first->d0()) < fabs(tracks[j].first->d0()))
                  keep[j] = false;
                else
                  keep[i] = false;
                changes++;

                LogTrace("TrackCleaner") << "   Sharing " << (*sharing).second << " remove by d0";
              }
            }
          }
        } else {  // pair tracks
          if ((*sharing).second > 0) {
            // Remove second track
            keep[j] = false;

            changes++;
          }
        }
      }
      /*
    }
    while(addedNewHit);
*/
    }
  } while (changes > 0);

  // Final copy
  TracksWithRecHits cleaned;

  for (unsigned int i = 0; i < tracks.size(); i++)
    if (keep[i])
      cleaned.push_back(tracks[i]);
    else
      delete tracks_[i].first;

  LogTrace("MinBiasTracking") << " [TrackCleaner] cleaned tracks : " << cleaned.size();

  for (unsigned int i = 0; i < cleaned.size(); i++)
    LogTrace("TrackCleaner") << "   Track #" << i << " : " << HitInfo::getInfo(cleaned[i].second, tTopo_);

  return cleaned;
}
