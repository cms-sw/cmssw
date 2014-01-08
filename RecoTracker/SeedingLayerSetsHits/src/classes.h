#include "RecoTracker/SeedingLayerSetsHits/interface/SeedingLayerSetsHits.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace RecoTracker_SeedingLayerSet {
  struct dictionary {
    edm::Wrapper<SeedingLayerSetsHits> wslsn;
  };
}
