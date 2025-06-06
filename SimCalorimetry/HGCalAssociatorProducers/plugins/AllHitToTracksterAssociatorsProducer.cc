// Author: Felice Pantaleo, felice.pantaleo@cern.ch 08/2024

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/Common/interface/MultiSpan.h"

class AllHitToTracksterAssociatorsProducer : public edm::global::EDProducer<> {
public:
  explicit AllHitToTracksterAssociatorsProducer(const edm::ParameterSet&);
  ~AllHitToTracksterAssociatorsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<ticl::Trackster>>>> tracksterCollectionTokens_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMapToken_;
  edm::EDGetTokenT<MultiHGCRecHitCollection> hitsToken_;
};

AllHitToTracksterAssociatorsProducer::AllHitToTracksterAssociatorsProducer(const edm::ParameterSet& pset)
    : layerClustersToken_(consumes<std::vector<reco::CaloCluster>>(pset.getParameter<edm::InputTag>("layerClusters"))),
      hitMapToken_(
          consumes<std::unordered_map<DetId, const unsigned int>>(pset.getParameter<edm::InputTag>("hitMapTag"))),
      hitsToken_(consumes<MultiHGCRecHitCollection>(pset.getParameter<edm::InputTag>("hits"))) {
  const auto& tracksterCollections = pset.getParameter<std::vector<edm::InputTag>>("tracksterCollections");
  for (const auto& tag : tracksterCollections) {
    tracksterCollectionTokens_.emplace_back(tag.label() + tag.instance(), consumes<std::vector<ticl::Trackster>>(tag));
  }

  for (const auto& tracksterToken : tracksterCollectionTokens_) {
    produces<ticl::AssociationMap<ticl::mapWithFraction>>("hitTo" + tracksterToken.first);
    produces<ticl::AssociationMap<ticl::mapWithFraction>>(tracksterToken.first + "ToHit");
  }
}

void AllHitToTracksterAssociatorsProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  using namespace edm;

  Handle<std::vector<reco::CaloCluster>> layer_clusters;
  iEvent.getByToken(layerClustersToken_, layer_clusters);

  if (!layer_clusters.isValid()) {
    edm::LogWarning("AllHitToTracksterAssociatorsProducer") << "Missing LayerCluster collection.";
    for (const auto& tracksterToken : tracksterCollectionTokens_) {
      iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(), "hitTo" + tracksterToken.first);
      iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(), tracksterToken.first + "ToHit");
    }
    return;
  }

  Handle<std::unordered_map<DetId, const unsigned int>> hitMap;
  iEvent.getByToken(hitMapToken_, hitMap);

  if (!iEvent.getHandle(hitsToken_)) {
    edm::LogWarning("AllHitToTracksterAssociatorsProducer") << "Missing MultiHGCRecHitCollection.";
    for (const auto& tracksterToken : tracksterCollectionTokens_) {
      iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(), "hitTo" + tracksterToken.first);
      iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(), tracksterToken.first + "ToHit");
    }
    return;
  }

  // Protection against missing HGCRecHitCollection
  const auto hits = iEvent.get(hitsToken_);
  for (const auto& hgcRecHitCollection : hits) {
    if (hgcRecHitCollection->empty()) {
      edm::LogWarning("AllHitToTracksterAssociatorsProducer") << "One of the HGCRecHitCollections is not valid.";
    }
  }

  edm::MultiSpan<HGCRecHit> rechitSpan(hits);
  // Check if rechitSpan is empty
  if (rechitSpan.size() == 0) {
    edm::LogWarning("HitToSimClusterCaloParticleAssociatorProducer")
        << "No valid HGCRecHitCollections found. Association maps will be empty.";
    for (const auto& tracksterToken : tracksterCollectionTokens_) {
      iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(), "hitTo" + tracksterToken.first);
      iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(), tracksterToken.first + "ToHit");
    }
    return;
  }

  for (const auto& tracksterToken : tracksterCollectionTokens_) {
    Handle<std::vector<ticl::Trackster>> tracksters;
    iEvent.getByToken(tracksterToken.second, tracksters);

    if (!tracksters.isValid()) {
      edm::LogWarning("AllHitToTracksterAssociatorsProducer") << "Missing Tracksters for one of the hitsTokens.";
      iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(), "hitTo" + tracksterToken.first);
      iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(), tracksterToken.first + "ToHit");
      continue;
    }

    auto hitToTracksterMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(rechitSpan.size());
    auto tracksterToHitMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(tracksters->size());

    for (unsigned int tracksterId = 0; tracksterId < tracksters->size(); ++tracksterId) {
      const auto& trackster = (*tracksters)[tracksterId];
      for (unsigned int j = 0; j < trackster.vertices().size(); ++j) {
        const auto& lc = (*layer_clusters)[trackster.vertices()[j]];
        float invMultiplicity = 1.0f / trackster.vertex_multiplicity()[j];

        for (const auto& hitAndFraction : lc.hitsAndFractions()) {
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

    iEvent.put(std::move(hitToTracksterMap), "hitTo" + tracksterToken.first);
    iEvent.put(std::move(tracksterToHitMap), tracksterToken.first + "ToHit");
  }
}

void AllHitToTracksterAssociatorsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("tracksterCollections",
                                       {edm::InputTag("ticlTrackstersCLUE3DHigh"),
                                        edm::InputTag("ticlTrackstersLinks"),
                                        edm::InputTag("ticlCandidate")});
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
  desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "MultiHGCRecHitCollectionProduct"));
  descriptions.add("AllHitToTracksterAssociatorsProducer", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(AllHitToTracksterAssociatorsProducer);
