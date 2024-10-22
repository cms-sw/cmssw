#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>

namespace RecoTracker_TkHitPairs {
  struct dictionary {
    IntermediateHitDoublets ihd;
    edm::Wrapper<IntermediateHitDoublets> wihd;

    RegionsSeedingHitSets rshs;
    edm::Wrapper<RegionsSeedingHitSets> wrshs;
  };
}  // namespace RecoTracker_TkHitPairs
