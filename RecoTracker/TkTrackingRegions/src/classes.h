#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionsSeedingLayerSetsHits.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Wrapper.h"


namespace RecoTracker_TkTrackingRegions {
  struct dictionary {
    edm::OwnVector<TrackingRegion> ovtr;
    edm::Wrapper<edm::OwnVector<TrackingRegion> > wovtr;

    edm::Wrapper<TrackingRegionsSeedingLayerSetsHits> wtrslsh;
  };
}
