#include "LCToSCAssociatorByEnergyScoreProducer.h"

#include <memory>

template <typename HIT, typename CLUSTER>
LCToSCAssociatorByEnergyScoreProducerT<HIT, CLUSTER>::LCToSCAssociatorByEnergyScoreProducerT(const edm::ParameterSet &ps)
    : hitMap_(consumes<std::unordered_map<DetId, const unsigned int>>(ps.getParameter<edm::InputTag>("hitMapTag"))),
      caloGeometry_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      hardScatterOnly_(ps.getParameter<bool>("hardScatterOnly")),
      hits_token_(consumes<multiCollectionT>(ps.getParameter<edm::InputTag>("hits"))) {
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

  if (!iEvent.getHandle(hitMap_) || !iEvent.getHandle(hits_token_)) {
    if (!iEvent.getHandle(hitMap_)) {
      edm::LogWarning("LCToSCAssociatorByEnergyScoreProducer") << "Hit map not valid. Producing empty associator.";
    }
    if (!iEvent.getHandle(hits_token_)) {
      edm::LogWarning("LCToSCAssociatorByEnergyScoreProducer")
          << "Hit MultiCollection not available. Producing empty associator.";
    }

    const std::unordered_map<DetId, const unsigned int> hitMap;  // empty map
    const multiCollectionT hits;
    auto impl = std::make_unique<LCToSCAssociatorByEnergyScoreImplT<HIT, CLUSTER>>(
        iEvent.productGetter(), hardScatterOnly_, rhtools_, &hitMap, hits);
    auto emptyAssociator = std::make_unique<ticl::LayerClusterToSimClusterAssociatorT<CLUSTER>>(std::move(impl));
    iEvent.put(std::move(emptyAssociator));
    return;
  }

  const auto hits = iEvent.get(hits_token_);
  const bool no_hits =
      std::none_of(hits.begin(), hits.end(), [](const auto &subCollection) { return !subCollection->empty(); });

  if (no_hits) {
    edm::LogWarning("LCToSCAssociatorByEnergyScoreProducerT") << "No hits collected. Producing empty associator.";
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
    desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "MultiHGCRecHitCollectionProduct"));
  } else {
    desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "barrelRecHitMap"));
    desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "MultiPFRecHitCollectionProduct"));
  }
  cfg.addWithDefaultLabel(desc);
}
