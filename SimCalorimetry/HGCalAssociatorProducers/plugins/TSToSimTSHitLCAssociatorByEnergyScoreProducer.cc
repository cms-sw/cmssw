// Original author: Leonardo Cristella

// user include files
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociator.h"
#include "TSToSimTSHitLCAssociatorByEnergyScoreImpl.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

template <typename HIT>
class TSToSimTSHitLCAssociatorByEnergyScoreProducer : public edm::global::EDProducer<> {
public:
  explicit TSToSimTSHitLCAssociatorByEnergyScoreProducer(const edm::ParameterSet &);
  ~TSToSimTSHitLCAssociatorByEnergyScoreProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMap_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_;
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> rhtools_;
  std::vector<edm::InputTag> hits_label_;
  std::vector<edm::EDGetTokenT<std::vector<HIT>>> hits_token_;
};

template <typename HIT>
TSToSimTSHitLCAssociatorByEnergyScoreProducer<HIT>::TSToSimTSHitLCAssociatorByEnergyScoreProducer(
    const edm::ParameterSet &ps)
    : hitMap_(consumes<std::unordered_map<DetId, const unsigned int>>(ps.getParameter<edm::InputTag>("hitMapTag"))),
      caloGeometry_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      hardScatterOnly_(ps.getParameter<bool>("hardScatterOnly")),
      hits_label_(ps.getParameter<std::vector<edm::InputTag>>("hits")) {
  rhtools_ = std::make_shared<hgcal::RecHitTools>();

  for (auto &label : hits_label_) {
    hits_token_.push_back(consumes<std::vector<HIT>>(label));
  }

  // Register the product
  produces<ticl::TracksterToSimTracksterHitLCAssociator>();
}

template <typename HIT>
TSToSimTSHitLCAssociatorByEnergyScoreProducer<HIT>::~TSToSimTSHitLCAssociatorByEnergyScoreProducer() {}

template <typename HIT>
void TSToSimTSHitLCAssociatorByEnergyScoreProducer<HIT>::produce(edm::StreamID,
                                                                 edm::Event &iEvent,
                                                                 const edm::EventSetup &es) const {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeometry_);
  rhtools_->setGeometry(*geom);

  std::vector<const HIT *> hits;
  for (auto &token : hits_token_) {
    edm::Handle<std::vector<HIT>> hits_handle;
    iEvent.getByToken(token, hits_handle);
    for (const auto &hit : *hits_handle) {
      hits.push_back(&hit);
    }
  }

  if (hits.empty()) {
    edm::LogWarning("TSToSimTSHitLCAssociatorByEnergyScoreProducer")
        << "No hits collected. Producing empty associator.";
  }

  if (!iEvent.getHandle(hitMap_)) {
    edm::LogWarning("TSToSimTSHitLCAssociatorByEnergyScoreProducer")
        << "Hit map not valid. Producing empty associator.";

    const std::unordered_map<DetId, const unsigned int> hitMap;  // empty map
    auto impl = std::make_unique<TSToSimTSHitLCAssociatorByEnergyScoreImpl<HIT>>(
        iEvent.productGetter(), hardScatterOnly_, rhtools_, &hitMap, hits);
    auto emptyAssociator = std::make_unique<ticl::TracksterToSimTracksterHitLCAssociator>(std::move(impl));
    iEvent.put(std::move(emptyAssociator));
    return;
  }

  const auto hitMap = &iEvent.get(hitMap_);
  auto impl = std::make_unique<TSToSimTSHitLCAssociatorByEnergyScoreImpl<HIT>>(
      iEvent.productGetter(), hardScatterOnly_, rhtools_, hitMap, hits);
  auto toPut = std::make_unique<ticl::TracksterToSimTracksterHitLCAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

template <typename HIT>
void TSToSimTSHitLCAssociatorByEnergyScoreProducer<HIT>::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
    desc.add<std::vector<edm::InputTag>>("hits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  } else {
    desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "barrelRecHitMap"));
    desc.add<std::vector<edm::InputTag>>(
        "hits", {edm::InputTag("particleFlowRecHitECAL"), edm::InputTag("particleFlowRecHitHBHE")});
  }
  desc.add<bool>("hardScatterOnly", true);

  if constexpr (std::is_same_v<HIT, HGCRecHit>)
    cfg.add("hgcalSimTracksterHitLCAssociatorByEnergyScore", desc);
  else
    cfg.add("barrelSimTracksterHitLCAssociatorByEnergyScore", desc);
}

//define this as a plug-in
template class TSToSimTSHitLCAssociatorByEnergyScoreProducer<HGCRecHit>;
using HGCalTSToSimTSHitLCAssociatorByEnergyScoreProducer = TSToSimTSHitLCAssociatorByEnergyScoreProducer<HGCRecHit>;
DEFINE_FWK_MODULE(HGCalTSToSimTSHitLCAssociatorByEnergyScoreProducer);
template class TSToSimTSHitLCAssociatorByEnergyScoreProducer<reco::PFRecHit>;
using BarrelTSToSimTSHitLCAssociatorByEnergyScoreProducer =
    TSToSimTSHitLCAssociatorByEnergyScoreProducer<reco::PFRecHit>;
DEFINE_FWK_MODULE(BarrelTSToSimTSHitLCAssociatorByEnergyScoreProducer);
