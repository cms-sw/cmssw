#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace TrackingTools_TransientTrackingRecHit {
  struct dictionary {
    edm::Wrapper<SeedingLayerSetsHits> wslsn;
  };
}
