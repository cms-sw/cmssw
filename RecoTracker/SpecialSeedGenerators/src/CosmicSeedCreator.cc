#include "RecoTracker/SpecialSeedGenerators/interface/CosmicSeedCreator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

namespace {
  template <class T>
  inline T sqr(T t) {
    return t * t;
  }
}  // namespace

void CosmicSeedCreator::init(const TrackingRegion& iregion, const edm::EventSetup& es, const SeedComparitor* ifilter) {
  region = &iregion;
  filter = ifilter;
  // mag field
  es.get<IdealMagneticFieldRecord>().get(bfield);
}

void CosmicSeedCreator::makeSeed(TrajectorySeedCollection& seedCollection, const SeedingHitSet& ordered) {
  //_________________________
  //
  //Get Parameters
  //________________________

  //hit package
  //+++++++++++
  const SeedingHitSet& hits = ordered;
  if (hits.size() < 2)
    return;

  //hits
  //++++
  SeedingHitSet::ConstRecHitPointer tth1 = hits[0];
  SeedingHitSet::ConstRecHitPointer tth2 = hits[1];
  assert(!trackerHitRTTI::isUndef(*tth1));
  assert(!trackerHitRTTI::isUndef(*tth2));

  SeedingHitSet::ConstRecHitPointer usedHit;

  //definition of position & momentum
  //++++++++++++++++++++++++++++++++++
  //direction of the trajectory seed given by the direction of the region
  GlobalVector initialMomentum(region->direction());
  //fix the momentum scale
  //initialMomentum = initialMomentum.basicVector.unitVector() * region->origin().direction().mag();
  //initialMomentum = region->origin().direction(); //alternative.
  LogDebug("CosmicSeedCreator") << "initial momentum = " << initialMomentum;

  //___________________________________________
  //
  //Direction of the trajectory seed
  //___________________________________________

  //radius
  //++++++
  bool reverseAll = false;
  if (fabs(tth1->globalPosition().perp()) < fabs(tth2->globalPosition().perp()))
  //comparison of the position of the 2 hits by checking/comparing their radius
  {
    usedHit = tth1;
    reverseAll = true;
  }

  else
    usedHit = tth2;

  //location in the barrel (up or bottom)
  //+++++++++++++++++++++++++++++++++++++
  //simple check, probably nees to be more precise FIXME
  bool bottomSeed = (usedHit->globalPosition().y() < 0);

  //apply corrections
  //+++++++++++++++++
  edm::OwnVector<TrackingRecHit> seedHits;

  if (reverseAll) {
    LogDebug("CosmicSeedCreator") << "Reverse all applied";

    seedHits.push_back(tth2->clone());
    seedHits.push_back(tth1->clone());
  }

  else {
    seedHits.push_back(tth1->clone());
    seedHits.push_back(tth2->clone());
  }

  //propagation
  //+++++++++++

  PropagationDirection seedDirection = alongMomentum;  //by default

  if (reverseAll)
    initialMomentum *= -1;

  if (bottomSeed) {
    //means that the seed parameters are inverse of what we want.
    //reverse the momentum again
    initialMomentum *= -1;
    //and change the direction of the seed
    seedDirection = oppositeToMomentum;
  }

  for (int charge = -1; charge <= 1; charge += 2) {
    //fixme, what hit do you want to use ?

    FreeTrajectoryState freeState(
        GlobalTrajectoryParameters(usedHit->globalPosition(), initialMomentum, charge, &*bfield),
        CurvilinearTrajectoryError(ROOT::Math::SMatrixIdentity()));

    LogDebug("CosmicSeedCreator") << "Position freeState: " << usedHit->globalPosition() << "\nCharge: " << charge
                                  << "\nInitial momentum :" << initialMomentum;

    TrajectoryStateOnSurface tsos(freeState, *usedHit->surface());

    PTrajectoryStateOnDet const& PTraj =
        trajectoryStateTransform::persistentState(tsos, usedHit->hit()->geographicalId().rawId());
    seedCollection.emplace_back(PTraj, seedHits, seedDirection);

  }  //end charge loop

  //________________
  //
  //Return seed
  //________________

  LogDebug("CosmicSeedCreator") << "Using SeedCreator---------->\n"
                                << "seedCollections size = " << seedCollection.size();

  if (seedCollection.size() > maxseeds_) {
    edm::LogError("TooManySeeds") << "Found too many seeds (" << seedCollection.size() << " > " << maxseeds_
                                  << "), bailing out.\n";
    seedCollection.clear();
  }
}
