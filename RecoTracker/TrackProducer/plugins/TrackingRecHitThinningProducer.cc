#include "RecoTracker/TrackProducer/plugins/TrackingRecHitThinningProducer.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

TrackingRecHitSelector::TrackingRecHitSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc)
    : trackExtraToken_(cc.consumes<reco::TrackExtraCollection>(pset.getParameter<edm::InputTag>("trackExtraTag"))) {}

void TrackingRecHitSelector::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("trackExtraTag");
}

void TrackingRecHitSelector::preChooseRefs(edm::Handle<TrackingRecHitCollection> hits,
                                           edm::Event const& event,
                                           edm::EventSetup const& es) {
  auto trackExtras = event.getHandle(trackExtraToken_);

  for (const auto& trackExtra : *trackExtras) {
    for (unsigned int i = 0; i < trackExtra.recHitsSize(); ++i) {
      addRef(trackExtra.recHit(i));
    }
  }
}
