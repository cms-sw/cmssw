// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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

  std::vector<edm::InputTag> hgcalRecHitTags_;
  std::vector<edm::InputTag> pfRecHitTags_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;
  // One warning per job per collection that is missing from the event.
  mutable std::vector<std::once_flag> hgcalMissingWarned_;
  mutable std::vector<std::once_flag> pfMissingWarned_;
};

DetIdToRecHitMapProducer::DetIdToRecHitMapProducer(edm::ParameterSet const& ps)
    : hgcalRecHitTags_(ps.getParameter<std::vector<edm::InputTag>>("hgcalRecHits")),
      pfRecHitTags_(ps.getParameter<std::vector<edm::InputTag>>("pfRecHits")),
      hgcalMissingWarned_(hgcalRecHitTags_.size()),
      pfMissingWarned_(pfRecHitTags_.size()) {
  hgcalRecHitTokens_.reserve(hgcalRecHitTags_.size());
  for (auto const& tag : hgcalRecHitTags_) {
    hgcalRecHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  }

  pfRecHitTokens_.reserve(pfRecHitTags_.size());
  for (auto const& tag : pfRecHitTags_) {
    pfRecHitTokens_.push_back(consumes<reco::PFRecHitCollection>(tag));
  }

  produces<hgcal::DetIdRecHitMap>();
}

void DetIdToRecHitMapProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  auto output = std::make_unique<hgcal::DetIdRecHitMap>();

  uint32_t globalRecHitIndex = 0;

  for (std::size_t i = 0; i < hgcalRecHitTokens_.size(); ++i) {
    edm::Handle<HGCRecHitCollection> handle;
    event.getByToken(hgcalRecHitTokens_[i], handle);

    if (!handle.isValid()) {
      std::call_once(hgcalMissingWarned_[i], [this, i]() {
        edm::LogWarning("DetIdToRecHitMapProducer") << "Missing HGCRecHitCollection " << hgcalRecHitTags_[i].encode()
                                                    << ". Its cells get no recHit index in this job.";
      });
      continue;
    }

    output->reserve(output->size() + handle->size());

    for (auto const& hit : *handle) {
      output->add(hit.detid().rawId(), globalRecHitIndex);
      ++globalRecHitIndex;
    }
  }

  for (std::size_t i = 0; i < pfRecHitTokens_.size(); ++i) {
    edm::Handle<reco::PFRecHitCollection> handle;
    event.getByToken(pfRecHitTokens_[i], handle);

    if (!handle.isValid()) {
      std::call_once(pfMissingWarned_[i], [this, i]() {
        edm::LogWarning("DetIdToRecHitMapProducer") << "Missing reco::PFRecHitCollection " << pfRecHitTags_[i].encode()
                                                    << ". Its cells get no recHit index in this job.";
      });
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

  // Barrel and forward PFRecHits, in the global-index order the consumers
  // (e.g. TruthLogicalGraphDumper) must mirror exactly. Keep this list and that
  // one in sync: the recHit index is pure concatenation order, so adding,
  // removing or reordering a collection shifts every downstream index.
  desc.add<std::vector<edm::InputTag>>("pfRecHits",
                                       {edm::InputTag("particleFlowRecHitECAL"),
                                        edm::InputTag("particleFlowRecHitHBHE"),
                                        edm::InputTag("particleFlowRecHitHF"),
                                        edm::InputTag("particleFlowRecHitHO")});

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(DetIdToRecHitMapProducer);
