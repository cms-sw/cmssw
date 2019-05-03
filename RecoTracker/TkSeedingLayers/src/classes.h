#include "DataFormats/Common/interface/Wrapper.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

#include <vector>

namespace RecoTracker_TkSeedingLayers {
struct dictionary {
  std::vector<SeedingHitSet> vshs;
  edm::Wrapper<std::vector<SeedingHitSet>> wvshs;
};
} // namespace RecoTracker_TkSeedingLayers
