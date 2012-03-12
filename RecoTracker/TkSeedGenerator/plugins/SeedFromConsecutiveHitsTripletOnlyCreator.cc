#include "SeedFromConsecutiveHitsTripletOnlyCreator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"


GlobalTrajectoryParameters SeedFromConsecutiveHitsTripletOnlyCreator::initialKinematic(
      const SeedingHitSet& hits, 
      const TrackingRegion & region, 
      const edm::EventSetup& es) const
{
  GlobalTrajectoryParameters kine;

  const TransientTrackingRecHit::ConstRecHitPointer& tth1 = hits[0];
  const TransientTrackingRecHit::ConstRecHitPointer& tth2 = hits[1];
  const TransientTrackingRecHit::ConstRecHitPointer& tth3 = hits[2];

  FastHelix helix(tth3->globalPosition(), tth2->globalPosition(), tth1->globalPosition(), es, tth1->globalPosition());
  kine = helix.stateAtVertex().parameters();

  edm::ESHandle<MagneticField> bfield;
  es.get<IdealMagneticFieldRecord>().get(bfield);
  bool isBOFF = ( std::abs(bfield->inTesla(GlobalPoint(0,0,0)).z()) < 1e-3 );
  if (isBOFF && (theBOFFMomentum > 0)) {
    kine = GlobalTrajectoryParameters(kine.position(),
                              kine.momentum().unit() * theBOFFMomentum,
                              kine.charge(),
                              &*bfield);
  }
  return kine;
}
