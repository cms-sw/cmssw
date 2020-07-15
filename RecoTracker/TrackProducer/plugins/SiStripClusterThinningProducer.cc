#include "RecoTracker/TrackProducer/plugins/SiStripClusterThinningProducer.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

SiStripClusterSelector::SiStripClusterSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc) {
  std::vector<edm::InputTag> trackingRecHitsInputTags =
      pset.getParameter<std::vector<edm::InputTag> >("trackingRecHitsTags");
  for (edm::InputTag const& tag : trackingRecHitsInputTags) {
    trackingRecHitsTokens_.push_back(cc.consumes<TrackingRecHitCollection>(tag));
  }
}

void SiStripClusterSelector::fillDescription(edm::ParameterSetDescription& desc) {
  desc.add<std::vector<edm::InputTag> >("trackingRecHitsTags");
}

void SiStripClusterSelector::preChooseRefs(edm::Handle<edmNew::DetSetVector<SiStripCluster> > hits,
                                           edm::Event const& event,
                                           edm::EventSetup const& es) {
  for (auto const& token : trackingRecHitsTokens_) {
    auto trackingRecHits = event.getHandle(token);

    for (const auto& hit : *trackingRecHits) {
      TrackerSingleRecHit const* singleHit = dynamic_cast<TrackerSingleRecHit const*>(&hit);
      if (singleHit != nullptr) {
        edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster> const& stripRef = singleHit->cluster_strip();
        if (stripRef.isNonnull()) {
          addRef(stripRef);
        }
        SiStripMatchedRecHit2D const* matched2DHit = dynamic_cast<SiStripMatchedRecHit2D const*>(singleHit);
        if (matched2DHit != nullptr) {
          edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster> const& monoRef =
              matched2DHit->monoClusterRef().cluster_strip();
          edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster> const& stereoRef =
              matched2DHit->stereoClusterRef().cluster_strip();
          if (monoRef.isNonnull()) {
            addRef(monoRef);
          }
          if (stereoRef.isNonnull()) {
            addRef(stereoRef);
          }
        }
      }
    }
  }
}
