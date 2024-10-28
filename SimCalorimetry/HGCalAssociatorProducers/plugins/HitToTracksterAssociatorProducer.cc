// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024

// user include files
#include "HitToTracksterAssociatorProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/Common/interface/MultiSpan.h"

template <typename HIT>
HitToTracksterAssociatorProducerT<HIT>::HitToTracksterAssociatorProducerT(const edm::ParameterSet &pset)
    : LCCollectionToken_(consumes<std::vector<reco::CaloCluster>>(pset.getParameter<edm::InputTag>("layer_clusters"))),
      tracksterCollectionToken_(consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("tracksters"))),
      hitMapToken_(
          consumes<std::unordered_map<DetId, const unsigned int>>(pset.getParameter<edm::InputTag>("hitMapTag"))) {
  auto hitsTags = pset.getParameter<std::vector<edm::InputTag>>("hits");
  for (const auto &tag : hitsTags) {
    hitsTokens_.push_back(consumes<std::vector<HIT>>(tag));
  }
  produces<ticl::AssociationMap<ticl::mapWithFraction>>("hitToTracksterMap");
  produces<ticl::AssociationMap<ticl::mapWithFraction>>("tracksterToHitMap");
}

template <typename HIT>
HitToTracksterAssociatorProducerT<HIT>::~HitToTracksterAssociatorProducerT() {}

template <typename HIT>
void HitToTracksterAssociatorProducerT<HIT>::produce(edm::StreamID,
                                                     edm::Event &iEvent,
                                                     const edm::EventSetup &iSetup) const {
  using namespace edm;

  Handle<std::vector<reco::CaloCluster>> layer_clusters;
  iEvent.getByToken(LCCollectionToken_, layer_clusters);

  Handle<std::vector<ticl::Trackster>> tracksters;
  iEvent.getByToken(tracksterCollectionToken_, tracksters);

  Handle<std::unordered_map<DetId, const unsigned int>> hitMap;
  iEvent.getByToken(hitMapToken_, hitMap);

  edm::MultiSpan<HIT> rechitSpan;
  for (const auto &token : hitsTokens_) {
    Handle<std::vector<HIT>> hitsHandle;
    iEvent.getByToken(token, hitsHandle);
    rechitSpan.add(*hitsHandle);
  }

  // Create association map
  auto hitToTracksterMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(rechitSpan.size());
  auto tracksterToHitMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(tracksters->size());

  // Loop over tracksters
  for (unsigned int tracksterId = 0; tracksterId < tracksters->size(); ++tracksterId) {
    const auto &trackster = (*tracksters)[tracksterId];
    // Loop over vertices in trackster
    for (unsigned int i = 0; i < trackster.vertices().size(); ++i) {
      // Get layerCluster
      const auto &lc = (*layer_clusters)[trackster.vertices()[i]];
      float invMultiplicity = 1.0f / trackster.vertex_multiplicity()[i];

      for (const auto &hitAndFraction : lc.hitsAndFractions()) {
        auto hitMapIter = hitMap->find(hitAndFraction.first);
        if (hitMapIter != hitMap->end()) {
          unsigned int rechitIndex = hitMapIter->second;
          float fraction = hitAndFraction.second * invMultiplicity;
          hitToTracksterMap->insert(rechitIndex, tracksterId, fraction);
          tracksterToHitMap->insert(tracksterId, rechitIndex, fraction);
        }
      }
    }
  }
  iEvent.put(std::move(hitToTracksterMap), "hitToTracksterMap");
  iEvent.put(std::move(tracksterToHitMap), "tracksterToHitMap");
}

template <typename HIT>
void HitToTracksterAssociatorProducerT<HIT>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTracksters"));
    desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
    desc.add<std::vector<edm::InputTag>>("hits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
    descriptions.add("hitToTracksterAssociator", desc);
  } else if constexpr (std::is_same_v<HIT, reco::PFRecHit>) {
    desc.add<edm::InputTag>("layer_clusters", edm::InputTag("barrelLayerClusters"));
    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlBarrelTracksters")),
        desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "barrelRecHitMap"));
    desc.add<std::vector<edm::InputTag>>(
        "hits", {edm::InputTag("particleFlowRecHitECAL"), edm::InputTag("particleFlowRecHitHBHE")});
    descriptions.add("hitToBarrelTracksterAssociator", desc);
  }
}

// Define this as a plug-in
using HitToTracksterAssociatorProducer = HitToTracksterAssociatorProducerT<HGCRecHit>;
DEFINE_FWK_MODULE(HitToTracksterAssociatorProducer);
using HitToBarrelTracksterAssociatorProducer = HitToTracksterAssociatorProducerT<reco::PFRecHit>;
DEFINE_FWK_MODULE(HitToBarrelTracksterAssociatorProducer);
