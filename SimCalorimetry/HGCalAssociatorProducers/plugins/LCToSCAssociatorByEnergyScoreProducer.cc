#include "LCToSCAssociatorByEnergyScoreProducer.h"

#include <memory>

template <typename HIT, typename CLUSTER>
LCToSCAssociatorByEnergyScoreProducerT<HIT, CLUSTER>::LCToSCAssociatorByEnergyScoreProducerT(const edm::ParameterSet &ps)
    : hitMap_(consumes<std::unordered_map<DetId, const unsigned int>>(ps.getParameter<edm::InputTag>("hitMapTag"))),
      caloGeometry_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      hardScatterOnly_(ps.getParameter<bool>("hardScatterOnly")),
      hits_label_(ps.getParameter<std::vector<edm::InputTag>>("hits")) {
  for (auto &label : hits_label_) {
    hits_token_.push_back(consumes<std::vector<HIT>>(label));
  }

  rhtools_ = std::make_shared<hgcal::RecHitTools>();

  // Register the product
  produces<ticl::LayerClusterToSimClusterAssociatorT<CLUSTER>>();
}

template <typename HIT, typename CLUSTER>
LCToSCAssociatorByEnergyScoreProducerT<HIT, CLUSTER>::~LCToSCAssociatorByEnergyScoreProducerT() {}

template <typename HIT, typename CLUSTER>
void LCToSCAssociatorByEnergyScoreProducerT<HIT, CLUSTER>::produce(edm::StreamID,
                                                                   edm::Event &iEvent,
                                                                   const edm::EventSetup &es) const {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeometry_);
  rhtools_->setGeometry(*geom);

  std::vector<const HIT *> hits;

  for (unsigned i = 0; i < hits_token_.size(); ++i) {
    auto hits_handle = iEvent.getHandle(hits_token_[i]);

    // Check handle validity
    if (!hits_handle.isValid()) {
      edm::LogWarning("LCToSCAssociatorByEnergyScoreProducer")
          << "Hit collection not available for token " << hits_label_[i] << ". Skipping this collection.";
      continue;  // Skip invalid handle
    }

    for (const auto &hit : *hits_handle) {
      hits.push_back(&hit);
    }
  }

  if (hits.empty()) {
    edm::LogWarning("LCToSCAssociatorByEnergyScoreProducerT") << "No hits collected. Producing empty associator.";
  }

  if (!iEvent.getHandle(hitMap_)) {
    edm::LogWarning("LCToSCAssociatorByEnergyScoreProducerT") << "Hit map not valid. Producing empty associator.";

    const std::unordered_map<DetId, const unsigned int> hitMap;  // empty map
    auto impl = std::make_unique<LCToSCAssociatorByEnergyScoreImplT<HIT, CLUSTER>>(
        iEvent.productGetter(), hardScatterOnly_, rhtools_, &hitMap, hits);
    auto emptyAssociator = std::make_unique<ticl::LayerClusterToSimClusterAssociatorT<CLUSTER>>(std::move(impl));
    iEvent.put(std::move(emptyAssociator));
    return;
  }

  const auto hitMap = &iEvent.get(hitMap_);
  auto impl = std::make_unique<LCToSCAssociatorByEnergyScoreImplT<HIT, CLUSTER>>(
      iEvent.productGetter(), hardScatterOnly_, rhtools_, hitMap, hits);
  auto toPut = std::make_unique<ticl::LayerClusterToSimClusterAssociatorT<CLUSTER>>(std::move(impl));
  iEvent.put(std::move(toPut));
}

template <typename HIT, typename CLUSTER>
void LCToSCAssociatorByEnergyScoreProducerT<HIT, CLUSTER>::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("hardScatterOnly", true);
  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
    desc.add<std::vector<edm::InputTag>>("hits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  } else {
    desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "barrelRecHitMap"));
    desc.add<std::vector<edm::InputTag>>(
        "hits", {edm::InputTag("particleFlowRecHitECAL", ""), edm::InputTag("particleFlowRecHitHBHE", "")});
  }
  cfg.addWithDefaultLabel(desc);
}
