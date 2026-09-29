// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include <cstdint>
#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "SimCalorimetry/HGCalAssociatorProducers/interface/DetIdRecHitMap.h"

class DetIdToRecHitMapProducer : public edm::global::EDProducer<> {
public:
  explicit DetIdToRecHitMapProducer(edm::ParameterSet const& ps);
  ~DetIdToRecHitMapProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;
};

DetIdToRecHitMapProducer::DetIdToRecHitMapProducer(edm::ParameterSet const& ps) {
  const auto& hgcalTags = ps.getParameter<std::vector<edm::InputTag>>("hgcalRecHits");
  const auto& pfTags = ps.getParameter<std::vector<edm::InputTag>>("pfRecHits");

  hgcalRecHitTokens_.reserve(hgcalTags.size());
  for (auto const& tag : hgcalTags) {
    hgcalRecHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  }

  pfRecHitTokens_.reserve(pfTags.size());
  for (auto const& tag : pfTags) {
    pfRecHitTokens_.push_back(consumes<reco::PFRecHitCollection>(tag));
  }

  produces<hgcal::DetIdRecHitMap>();
}

void DetIdToRecHitMapProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  auto output = std::make_unique<hgcal::DetIdRecHitMap>();

  uint32_t globalRecHitIndex = 0;

  for (auto const& token : hgcalRecHitTokens_) {
    edm::Handle<HGCRecHitCollection> handle;
    event.getByToken(token, handle);

    if (!handle.isValid()) {
      edm::LogWarning("DetIdToRecHitMapProducer") << "Missing HGCRecHitCollection. Skipping it.";
      continue;
    }

    output->reserve(output->size() + handle->size());

    for (auto const& hit : *handle) {
      output->add(hit.detid().rawId(), globalRecHitIndex);
      ++globalRecHitIndex;
    }
  }

  for (auto const& token : pfRecHitTokens_) {
    edm::Handle<reco::PFRecHitCollection> handle;
    event.getByToken(token, handle);

    if (!handle.isValid()) {
      edm::LogWarning("DetIdToRecHitMapProducer") << "Missing reco::PFRecHitCollection. Skipping it.";
      continue;
    }

    output->reserve(output->size() + handle->size());

    for (auto const& hit : *handle) {
      output->add(hit.detId(), globalRecHitIndex);
      ++globalRecHitIndex;
    }
  }

  // Sort for binary-search lookup and drop duplicate detIds (keeping the first
  // inserted index, as the previous hash-map build did).
  const uint32_t duplicates = output->finalize();
  if (duplicates > 0) {
    edm::LogWarning("DetIdToRecHitMapProducer")
        << duplicates
        << " duplicate DetId(s) across the configured RecHit collections; kept the first recHit index for each "
           "(check for overlapping inputs, e.g. HGCalRecHit mixed with particleFlowRecHitHGC).";
  }

  event.put(std::move(output));
}

void DetIdToRecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::vector<edm::InputTag>>("hgcalRecHits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});

  // Cleaned barrel/forward PFRecHits, in the global-index order the consumers
  // (e.g. TruthLogicalGraphDumper) must mirror exactly. Keep this list and that
  // one in sync: the recHit index is pure concatenation order, so adding,
  // removing or reordering a collection shifts every downstream index.
  desc.add<std::vector<edm::InputTag>>("pfRecHits",
                                       {edm::InputTag("particleFlowRecHitECAL", "Cleaned"),
                                        edm::InputTag("particleFlowRecHitHBHE", "Cleaned"),
                                        edm::InputTag("particleFlowRecHitHF", "Cleaned"),
                                        edm::InputTag("particleFlowRecHitHO", "Cleaned")});

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(DetIdToRecHitMapProducer);
