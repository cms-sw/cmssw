#ifndef RecoTracker_TkSeedGenerator_SeedFromProtoTrack_H
#define RecoTracker_TkSeedGenerator_SeedFromProtoTrack_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

namespace reco {
  class Track;
}
namespace edm {
  class EventSetup;
}

class SeedFromProtoTrack {
public:
  SeedFromProtoTrack(const reco::Track& proto, const edm::EventSetup&);
  SeedFromProtoTrack(const reco::Track& proto, const SeedingHitSet& hits, const edm::EventSetup& es);

  ~SeedFromProtoTrack() {}

  TrajectorySeed trajectorySeed() const;

  bool isValid() const { return theValid; }

private:
  void init(const reco::Track& proto, const edm::EventSetup& es);

  PropagationDirection direction() const { return alongMomentum; }

  PTrajectoryStateOnDet const& trajectoryState() const { return thePTraj; }

  typedef edm::OwnVector<TrackingRecHit> RecHitContainer;
  const RecHitContainer& hits() const { return theHits; }

private:
  bool theValid;
  RecHitContainer theHits;
  PTrajectoryStateOnDet thePTraj;
};
#endif
