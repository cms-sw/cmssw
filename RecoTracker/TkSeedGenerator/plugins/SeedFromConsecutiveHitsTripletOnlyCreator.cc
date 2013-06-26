#include "SeedFromConsecutiveHitsTripletOnlyCreator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"


bool SeedFromConsecutiveHitsTripletOnlyCreator::initialKinematic(GlobalTrajectoryParameters & kine,
								 const SeedingHitSet & hits) const {


  const TransientTrackingRecHit::ConstRecHitPointer& tth1 = hits[0];
  const TransientTrackingRecHit::ConstRecHitPointer& tth2 = hits[1];
  const TransientTrackingRecHit::ConstRecHitPointer& tth3 = hits[2];

  FastHelix helix(tth3->globalPosition(), tth2->globalPosition(), tth1->globalPosition(), nomField, &*bfield, tth1->globalPosition());
  kine = helix.stateAtVertex();

  if unlikely(isBOFF && (theBOFFMomentum > 0)) {
      kine = GlobalTrajectoryParameters(kine.position(),
					kine.momentum().unit() * theBOFFMomentum,
					kine.charge(),
					&*bfield);
  }
  return (filter ? filter->compatible(hits, kine, helix, *region) : true); 
}
