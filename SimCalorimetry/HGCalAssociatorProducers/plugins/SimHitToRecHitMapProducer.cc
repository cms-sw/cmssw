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

class SimHitToRecHitMapProducer : public edm::global::EDProducer<> {
public:
  explicit SimHitToRecHitMapProducer(edm::ParameterSet const& ps);
  ~SimHitToRecHitMapProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;
};

SimHitToRecHitMapProducer::SimHitToRecHitMapProducer(edm::ParameterSet const& ps) {
  const auto hgcalTags = ps.getParameter<std::vector<edm::InputTag>>("hgcalRecHits");
  const auto pfTags = ps.getParameter<std::vector<edm::InputTag>>("pfRecHits");

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

void SimHitToRecHitMapProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  auto output = std::make_unique<hgcal::DetIdRecHitMap>();

  uint32_t globalRecHitIndex = 0;

  for (auto const& token : hgcalRecHitTokens_) {
    edm::Handle<HGCRecHitCollection> handle;
    event.getByToken(token, handle);

    if (!handle.isValid()) {
      edm::LogWarning("SimHitToRecHitMapProducer") << "Missing HGCRecHitCollection. Skipping it.";
      continue;
    }

    output->reserve(output->size() + handle->size());

    for (auto const& hit : *handle) {
      const uint32_t rawId = hit.detid().rawId();

      const auto [_, inserted] = output->emplace(rawId, globalRecHitIndex);
      if (!inserted) {
        edm::LogWarning("SimHitToRecHitMapProducer")
            << "Duplicate HGCAL DetId rawId=" << rawId << ". Keeping the first recHit index.";
      }

      ++globalRecHitIndex;
    }
  }

  for (auto const& token : pfRecHitTokens_) {
    edm::Handle<reco::PFRecHitCollection> handle;
    event.getByToken(token, handle);

    if (!handle.isValid()) {
      edm::LogWarning("SimHitToRecHitMapProducer") << "Missing reco::PFRecHitCollection. Skipping it.";
      continue;
    }

    output->reserve(output->size() + handle->size());

    for (auto const& hit : *handle) {
      const uint32_t rawId = hit.detId();

      const auto [_, inserted] = output->emplace(rawId, globalRecHitIndex);
      if (!inserted) {
        edm::LogWarning("SimHitToRecHitMapProducer")
            << "Duplicate PFRecHit DetId rawId=" << rawId << ". Keeping the first recHit index.";
      }

      ++globalRecHitIndex;
    }
  }

  event.put(std::move(output));
}

void SimHitToRecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::vector<edm::InputTag>>("hgcalRecHits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});

  desc.add<std::vector<edm::InputTag>>("pfRecHits",
                                       {edm::InputTag("particleFlowRecHitECAL"),
                                        edm::InputTag("particleFlowRecHitHBHE"),
                                        edm::InputTag("particleFlowRecHitHO")});

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SimHitToRecHitMapProducer);