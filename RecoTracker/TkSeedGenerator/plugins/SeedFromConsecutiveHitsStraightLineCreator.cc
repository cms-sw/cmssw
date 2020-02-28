#include "SeedFromConsecutiveHitsStraightLineCreator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

bool SeedFromConsecutiveHitsStraightLineCreator::initialKinematic(GlobalTrajectoryParameters& kine,
                                                                  const SeedingHitSet& hits) const {
  const SeedingHitSet::ConstRecHitPointer& tth1 = hits[0];
  const SeedingHitSet::ConstRecHitPointer& tth2 = hits[1];

  const GlobalPoint& vertexPos = region->origin();

  // Assume initial state is straight line passing through beam spot
  // with direction given by innermost two seed hits (with big uncertainty)
  GlobalVector initMomentum(tth2->globalPosition() - tth1->globalPosition());
  double rescale = 1000. / initMomentum.perp();
  initMomentum *= rescale;  // set to approximately infinite momentum
  TrackCharge q = 1;        // irrelevant, since infinite momentum
  kine = GlobalTrajectoryParameters(vertexPos, initMomentum, q, &*bfield);

  return true;
}
