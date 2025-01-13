#include "LCToCPAssociatorByEnergyScoreProducer.h"

#include <memory>

template <typename HIT>
LCToCPAssociatorByEnergyScoreProducer<HIT>::LCToCPAssociatorByEnergyScoreProducer(const edm::ParameterSet &ps)
    : hitMap_(consumes<std::unordered_map<DetId, const unsigned int>>(ps.getParameter<edm::InputTag>("hitMapTag"))),
      caloGeometry_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      hardScatterOnly_(ps.getParameter<bool>("hardScatterOnly")),
      hits_label_(ps.getParameter<std::vector<edm::InputTag>>("hits")) {
  for (auto &label : hits_label_) {
    if constexpr (std::is_same_v<HIT, HGCRecHit>)
      hgcal_hits_token_.push_back(consumes<HGCRecHitCollection>(label));
    else
      hits_token_.push_back(consumes<std::vector<HIT>>(label));
  }

  rhtools_ = std::make_shared<hgcal::RecHitTools>();

  // Register the product
  produces<ticl::LayerClusterToCaloParticleAssociator>();
}

template <typename HIT>
LCToCPAssociatorByEnergyScoreProducer<HIT>::~LCToCPAssociatorByEnergyScoreProducer() {}

template <typename HIT>
void LCToCPAssociatorByEnergyScoreProducer<HIT>::produce(edm::StreamID,
                                                         edm::Event &iEvent,
                                                         const edm::EventSetup &es) const {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeometry_);
  rhtools_->setGeometry(*geom);

  std::vector<const HIT *> hits;
  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    for (auto &token : hgcal_hits_token_) {
      edm::Handle<HGCRecHitCollection> hits_handle;
      iEvent.getByToken(token, hits_handle);
      for (const auto &hit : *hits_handle) {
        hits.push_back(&hit);
      }
    }
  } else {
    for (auto &token : hits_token_) {
      edm::Handle<std::vector<HIT>> hits_handle;
      iEvent.getByToken(token, hits_handle);
      for (const auto &hit : *hits_handle) {
        hits.push_back(&hit);
      }
    }
  }

  const auto hitMap = &iEvent.get(hitMap_);

  auto impl = std::make_unique<LCToCPAssociatorByEnergyScoreImpl<HIT>>(
      iEvent.productGetter(), hardScatterOnly_, rhtools_, hitMap, hits);
  auto toPut = std::make_unique<ticl::LayerClusterToCaloParticleAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

template <typename HIT>
void LCToCPAssociatorByEnergyScoreProducer<HIT>::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
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
    desc.add<std::vector<edm::InputTag>>("hits",
                                         {edm::InputTag("particleFlowRecHitECAL", ""),
                                          edm::InputTag("particleFlowRecHitHBHE", ""),
                                          edm::InputTag("particleFlowRecHitHO", "")});
  }
  cfg.addWithDefaultLabel(desc);
}
