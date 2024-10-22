#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsStraightLineCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsStraightLineCreator_H
#include "FWCore/Utilities/interface/Visibility.h"

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "SeedFromConsecutiveHitsCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
class FreeTrajectoryState;

class dso_hidden SeedFromConsecutiveHitsStraightLineCreator final : public SeedFromConsecutiveHitsCreator {
public:
  SeedFromConsecutiveHitsStraightLineCreator(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : SeedFromConsecutiveHitsCreator(cfg, std::move(iC)) {}

  ~SeedFromConsecutiveHitsStraightLineCreator() override {}

private:
  bool initialKinematic(GlobalTrajectoryParameters& kine, const SeedingHitSet& hits) const override;
};
#endif
