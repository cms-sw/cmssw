#ifndef RecoTracker_TkSeedGenerator_CosmicSeedCreator_H
#define RecoTracker_TkSeedGenerator_CosmicSeedCreator_H

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

class CosmicSeedCreator final : public SeedCreator {
public:
  CosmicSeedCreator(const edm::ParameterSet &extra, edm::ConsumesCollector &&);

  ~CosmicSeedCreator() override {}

  // initialize the "event dependent state"
  void init(const TrackingRegion &region, const edm::EventSetup &es, const SeedComparitor *filter) override;

  // make job
  // fill seedCollection with the "TrajectorySeed"
  void makeSeed(TrajectorySeedCollection &seedCollection, const SeedingHitSet &hits) override;

private:
  const TrackingRegion *region = nullptr;
  const SeedComparitor *filter = nullptr;
  const MagneticField *bfield = nullptr;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldESToken_;

  unsigned int maxseeds_;
};
#endif
