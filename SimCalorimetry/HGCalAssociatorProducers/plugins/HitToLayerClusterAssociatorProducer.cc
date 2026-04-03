// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024

// user include files
#include "HitToLayerClusterAssociatorProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/Common/interface/MultiSpan.h"

template <typename HIT>
HitToLayerClusterAssociatorProducerT<HIT>::HitToLayerClusterAssociatorProducerT(const edm::ParameterSet &pset)
    : LCCollectionToken_(consumes<std::vector<reco::CaloCluster>>(pset.getParameter<edm::InputTag>("layer_clusters"))),
      hitMapToken_(consumes<std::unordered_map<DetId, unsigned int>>(pset.getParameter<edm::InputTag>("hitMap"))) {
  auto hitsTags = pset.getParameter<std::vector<edm::InputTag>>("hits");
  for (const auto &tag : hitsTags) {
    hitsTokens_.push_back(consumes<std::vector<HIT>>(tag));
  }
  produces<ticl::AssociationMap<ticl::mapWithFraction>>("hitToLayerClusterMap");
}

template <typename HIT>
HitToLayerClusterAssociatorProducerT<HIT>::~HitToLayerClusterAssociatorProducerT() {}

template <typename HIT>
void HitToLayerClusterAssociatorProducerT<HIT>::produce(edm::StreamID,
                                                        edm::Event &iEvent,
                                                        const edm::EventSetup &iSetup) const {
  using namespace edm;

  Handle<std::vector<reco::CaloCluster>> layer_clusters;
  iEvent.getByToken(LCCollectionToken_, layer_clusters);

  Handle<std::unordered_map<DetId, unsigned int>> hitMap;
  iEvent.getByToken(hitMapToken_, hitMap);

  edm::MultiSpan<HIT> rechitSpan;
  for (const auto &token : hitsTokens_) {
    Handle<std::vector<HIT>> hitsHandle;
    iEvent.getByToken(token, hitsHandle);
    rechitSpan.add(*hitsHandle);
  }

  // Create association map
  auto hitToLayerClusterMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(rechitSpan.size());

  // Loop over layer clusters
  for (unsigned int lcId = 0; lcId < layer_clusters->size(); ++lcId) {
    const auto &layer_cluster = (*layer_clusters)[lcId];

    for (const auto &hitAndFraction : layer_cluster.hitsAndFractions()) {
      auto hitMapIter = hitMap->find(hitAndFraction.first);
      if (hitMapIter != hitMap->end()) {
        unsigned int rechitIndex = hitMapIter->second;
        float fraction = hitAndFraction.second;
        hitToLayerClusterMap->insert(rechitIndex, lcId, fraction);
      }
    }
  }
  iEvent.put(std::move(hitToLayerClusterMap), "hitToLayerClusterMap");
}

template <typename HIT>
void HitToLayerClusterAssociatorProducerT<HIT>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<edm::InputTag>("hitMap", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
    desc.add<std::vector<edm::InputTag>>("hits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
    descriptions.add("hitToLayerClusterAssociator", desc);
  } else if constexpr (std::is_same_v<HIT, reco::PFRecHit>) {
    desc.add<edm::InputTag>("layer_clusters", edm::InputTag("barrelLayerClusters"));
    desc.add<edm::InputTag>("hitMap", edm::InputTag("recHitMapProducer", "barrelRecHitMap"));
    desc.add<std::vector<edm::InputTag>>(
        "hits", {edm::InputTag("particleFlowRecHitECAL"), edm::InputTag("particleFlowRecHitHBHE")});
    descriptions.add("hitToBarrelLayerClusterAssociator", desc);
  }
}
