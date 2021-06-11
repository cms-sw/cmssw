#include "SeedFromConsecutiveHitsTripletOnlyCreator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "FWCore/Utilities/interface/Likely.h"

bool SeedFromConsecutiveHitsTripletOnlyCreator::initialKinematic(GlobalTrajectoryParameters& kine,
                                                                 const SeedingHitSet& hits) const {
  SeedingHitSet::ConstRecHitPointer tth1 = hits[0];
  SeedingHitSet::ConstRecHitPointer tth2 = hits[1];

  if (hits.size() == 3 &&
      !(trackerHitRTTI::isSingleType(*hits[2]) && (hits[2]->geographicalId().subdetId() == SiStripDetId::TID ||
                                                   hits[2]->geographicalId().subdetId() == SiStripDetId::TEC))) {
    //if 3rd hit is mono and endcap pT is not well defined so take initial state from pair
    SeedingHitSet::ConstRecHitPointer tth3 = hits[2];
    FastHelix helix(tth3->globalPosition(),
                    tth2->globalPosition(),
                    tth1->globalPosition(),
                    nomField,
                    &*bfield,
                    tth1->globalPosition());
    kine = helix.stateAtVertex();
    if UNLIKELY (isBOFF && (theBOFFMomentum > 0)) {
      kine = GlobalTrajectoryParameters(
          kine.position(), kine.momentum().unit() * theBOFFMomentum, kine.charge(), &*bfield);
    }
    return (filter ? filter->compatible(hits, kine, helix) : true);
  }

  const GlobalPoint& vertexPos = region->origin();

  FastHelix helix(tth2->globalPosition(), tth1->globalPosition(), vertexPos, nomField, &*bfield);
  if (helix.isValid()) {
    kine = helix.stateAtVertex();
  } else {
    GlobalVector initMomentum(tth2->globalPosition() - vertexPos);
    initMomentum *= (100. / initMomentum.perp());
    kine = GlobalTrajectoryParameters(vertexPos, initMomentum, 1, &*bfield);
  }

  if UNLIKELY (isBOFF && (theBOFFMomentum > 0)) {
    kine =
        GlobalTrajectoryParameters(kine.position(), kine.momentum().unit() * theBOFFMomentum, kine.charge(), &*bfield);
  }
  return (filter ? filter->compatible(hits, kine, helix) : true);
}
