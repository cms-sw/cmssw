#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>

namespace RecoTracker_TkHitPairs {
  struct dictionary {
    IntermediateHitDoublets ihd;
    edm::Wrapper<IntermediateHitDoublets> wihd;
  };
}
