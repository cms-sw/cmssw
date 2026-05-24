// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024
#include "FWCore/Framework/interface/global/EDProducer.h"
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
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/Common/interface/MultiSpan.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

template <typename HIT>
class HitToSimClusterCaloParticleAssociatorProducerT : public edm::global::EDProducer<> {
public:
  using multiCollectionT = edm::RefProdVector<std::vector<HIT>>;

  explicit HitToSimClusterCaloParticleAssociatorProducerT(const edm::ParameterSet &);
  ~HitToSimClusterCaloParticleAssociatorProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  const edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;

  const edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMapToken_;
  edm::EDGetTokenT<multiCollectionT> hitsToken_;
};

template <typename HIT>
HitToSimClusterCaloParticleAssociatorProducerT<HIT>::HitToSimClusterCaloParticleAssociatorProducerT(
    const edm::ParameterSet &pset)
    : simClusterToken_(consumes<std::vector<SimCluster>>(pset.getParameter<edm::InputTag>("simClusters"))),
      hitMapToken_(consumes<std::unordered_map<DetId, const unsigned int>>(pset.getParameter<edm::InputTag>("hitMap"))),
      hitsToken_(consumes<multiCollectionT>(pset.getParameter<edm::InputTag>("hits"))) {
  produces<ticl::AssociationMap<ticl::mapWithFraction>>();
}

template <typename HIT>
void HitToSimClusterCaloParticleAssociatorProducerT<HIT>::produce(edm::StreamID,
                                                                  edm::Event &iEvent,
                                                                  const edm::EventSetup &iSetup) const {
  using namespace edm;

  std::vector<SimCluster> simClusters = iEvent.get(simClusterToken_);
  Handle<std::unordered_map<DetId, const unsigned int>> hitMap;
  iEvent.getByToken(hitMapToken_, hitMap);

  if (!iEvent.getHandle(hitsToken_).isValid()) {
    edm::LogWarning("HitToSimClusterCaloParticleAssociatorProducer")
        << "RecHitCollections is invalid.  Association maps will be empty.";
    // Store empty maps in the event
    iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>());
    return;
  }

  // Protection against missing HitCollection
  const auto hits = iEvent.get(hitsToken_);
  for (std::size_t index = 0; const auto &hitCollection : hits) {
    if (hitCollection->empty()) {
      LogDebug("HitToSimClusterCaloParticleAssociatorProducer") << "HitCollection #" << index << " is empty.";
    }
    index++;
  }

  edm::MultiSpan<HIT> rechitSpan(hits);
  // Check if rechitSpan is empty after processing hitsTokens_
  if (rechitSpan.size() == 0) {
    LogDebug("HitToSimClusterCaloParticleAssociatorProducer")
        << "RecHitCollection is empty. Association maps will be empty.";
    // Store empty maps in the event
    iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>());
    return;
  }

  // Create association maps
  auto hitToSimClusterMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(rechitSpan.size());

  for (std::size_t scId = 0; scId < simClusters.size(); ++scId) {
    // Loop over hits in simCluster
    for (const auto &hitAndFraction : simClusters[scId].hits_and_fractions()) {
      auto hitMapIter = hitMap->find(hitAndFraction.first);
      if (hitMapIter != hitMap->end()) {
        unsigned int rechitIndex = hitMapIter->second;
        float fraction = hitAndFraction.second;
        hitToSimClusterMap->insert(rechitIndex, scId, fraction);
      }
    }
  }
  iEvent.put(std::move(hitToSimClusterMap));
}

template <typename HIT>
void HitToSimClusterCaloParticleAssociatorProducerT<HIT>::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));

  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    desc.add<edm::InputTag>("hitMap", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
    desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "RefProdVectorHGCRecHitCollection"));
    descriptions.add("hitToSimClusterCaloParticleAssociator", desc);
  } else if constexpr (std::is_same_v<HIT, reco::PFRecHit>) {
    desc.add<edm::InputTag>("hitMap", edm::InputTag("recHitMapProducer", "barrelRecHitMap"));
    desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "RefProdVectorPFRecHitCollection"));
    descriptions.add("barrelHitToSimClusterCaloParticleAssociator", desc);
  }
}

// Define this as a plug-in
using HitToSimClusterCaloParticleAssociatorProducer = HitToSimClusterCaloParticleAssociatorProducerT<HGCRecHit>;
DEFINE_FWK_MODULE(HitToSimClusterCaloParticleAssociatorProducer);
using BarrelHitToSimClusterCaloParticleAssociatorProducer =
    HitToSimClusterCaloParticleAssociatorProducerT<reco::PFRecHit>;
DEFINE_FWK_MODULE(BarrelHitToSimClusterCaloParticleAssociatorProducer);
