#ifndef RecoTracker_TkSeedGenerator_SeedFromProtoTrack_H
#define RecoTracker_TkSeedGenerator_SeedFromProtoTrack_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include <boost/shared_ptr.hpp>

namespace reco { class Track; }
namespace edm { class EventSetup; }

class SeedFromProtoTrack {
public:


  SeedFromProtoTrack(const reco::Track & proto, const edm::EventSetup& ); 

  ~SeedFromProtoTrack() {}

  TrajectorySeed trajectorySeed() const;

  bool isValid() const { return theValid; }

private:

  PropagationDirection direction() const { return alongMomentum; }

  PTrajectoryStateOnDet trajectoryState() const { return *thePTraj; }


  typedef edm::OwnVector<TrackingRecHit> RecHitContainer;
  const RecHitContainer & hits() const { return theHits; }

private:

  bool theValid;
  RecHitContainer theHits;
  boost::shared_ptr<PTrajectoryStateOnDet> thePTraj;

};
#endif
