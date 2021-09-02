#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsTripletOnlyCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsTripletOnlyCreator_H
#include "FWCore/Utilities/interface/Visibility.h"

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "SeedFromConsecutiveHitsCreator.h"

class dso_hidden SeedFromConsecutiveHitsTripletOnlyCreator final : public SeedFromConsecutiveHitsCreator {
public:
  SeedFromConsecutiveHitsTripletOnlyCreator(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : SeedFromConsecutiveHitsCreator(cfg, std::move(iC)) {}

  static void fillDescriptions(edm::ParameterSetDescription& desc) {
    SeedFromConsecutiveHitsCreator::fillDescriptions(desc);
  }
  static const char* fillDescriptionsLabel() { return "ConsecutiveHitsTripletOnly"; }

  ~SeedFromConsecutiveHitsTripletOnlyCreator() override {}

private:
  bool initialKinematic(GlobalTrajectoryParameters& kine, const SeedingHitSet& hits) const override;
};
#endif
