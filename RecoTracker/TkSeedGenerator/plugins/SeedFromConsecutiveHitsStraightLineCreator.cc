#include "SeedFromConsecutiveHitsStraightLineCreator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"


GlobalTrajectoryParameters SeedFromConsecutiveHitsStraightLineCreator::initialKinematic(
      const SeedingHitSet & hits,
      const TrackingRegion & region,
      const edm::EventSetup& es) const
{
  GlobalTrajectoryParameters kine;

  const TransientTrackingRecHit::ConstRecHitPointer& tth1 = hits[0];
  const TransientTrackingRecHit::ConstRecHitPointer& tth2 = hits[1];

  const GlobalPoint& vertexPos = region.origin();
  edm::ESHandle<MagneticField> bfield;
  es.get<IdealMagneticFieldRecord>().get(bfield);

  // Assume initial state is straight line passing through beam spot
  // with direction given by innermost two seed hits (with big uncertainty)
  GlobalVector initMomentum(tth2->globalPosition() - tth1->globalPosition());
  double rescale = 1000./initMomentum.perp();
  initMomentum *= rescale; // set to approximately infinite momentum
  TrackCharge q = 1; // irrelevant, since infinite momentum
  kine = GlobalTrajectoryParameters(vertexPos, initMomentum, q, &*bfield);

  return kine;
}

