#include "RecoTracker/TrackProducer/plugins/SiPixelClusterThinningProducer.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"

SiPixelClusterSelector::SiPixelClusterSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc) {
  std::vector<edm::InputTag> trackingRecHitsInputTags =
      pset.getParameter<std::vector<edm::InputTag> >("trackingRecHitsTags");
  for (edm::InputTag const& tag : trackingRecHitsInputTags) {
    trackingRecHitsTokens_.push_back(cc.consumes<TrackingRecHitCollection>(tag));
  }
}

void SiPixelClusterSelector::fillDescription(edm::ParameterSetDescription& desc) {
  desc.add<std::vector<edm::InputTag> >("trackingRecHitsTags");
}

void SiPixelClusterSelector::preChooseRefs(edm::Handle<edmNew::DetSetVector<SiPixelCluster> > hits,
                                           edm::Event const& event,
                                           edm::EventSetup const& es) {
  for (auto const& token : trackingRecHitsTokens_) {
    auto trackingRecHits = event.getHandle(token);

    for (const auto& hit : *trackingRecHits) {
      TrackerSingleRecHit const* singleHit = dynamic_cast<TrackerSingleRecHit const*>(&hit);
      if (singleHit != nullptr) {
        edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& pixelRef = singleHit->cluster_pixel();
        if (pixelRef.isNonnull()) {
          addRef(pixelRef);
        }
      }
    }
  }
}
