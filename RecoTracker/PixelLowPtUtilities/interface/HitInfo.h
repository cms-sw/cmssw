#ifndef RecoTracker_PixelLowPtUtilities_HitInfo_h
#define RecoTracker_PixelLowPtUtilities_HitInfo_h

class DetId;
class TrackingRecHit;
class PSimHit;

#include <vector>
#include <string>

class TrackerTopology;

class HitInfo {
public:
  HitInfo();
  ~HitInfo();

  static std::string getInfo(const DetId &id, const TrackerTopology *tTopo);
  static std::string getInfo(const TrackingRecHit &recHit, const TrackerTopology *tTopo);
  static std::string getInfo(const std::vector<const TrackingRecHit *> &recHits, const TrackerTopology *tTopo);
  static std::string getInfo(const PSimHit &simHit, const TrackerTopology *tTopo);
};

#endif
