// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024
#include "HitToSimClusterCaloParticleAssociatorProducer.h"
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
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

HitToSimClusterCaloParticleAssociatorProducer::HitToSimClusterCaloParticleAssociatorProducer(
    const edm::ParameterSet &pset)
    : simClusterToken_(consumes<std::vector<SimCluster>>(pset.getParameter<edm::InputTag>("simClusters"))),
      caloParticleToken_(consumes<std::vector<CaloParticle>>(pset.getParameter<edm::InputTag>("caloParticles"))),
      hitMapToken_(
          consumes<std::unordered_map<DetId, const unsigned int>>(pset.getParameter<edm::InputTag>("hitMap"))) {
  auto hitsTags = pset.getParameter<std::vector<edm::InputTag>>("hits");
  for (const auto &tag : hitsTags) {
    hitsTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  }
  produces<ticl::AssociationMap<ticl::mapWithFraction>>("hitToSimClusterMap");
  produces<ticl::AssociationMap<ticl::mapWithFraction>>("hitToCaloParticleMap");
}

HitToSimClusterCaloParticleAssociatorProducer::~HitToSimClusterCaloParticleAssociatorProducer() {}

void HitToSimClusterCaloParticleAssociatorProducer::produce(edm::StreamID,
                                                            edm::Event &iEvent,
                                                            const edm::EventSetup &iSetup) const {
  using namespace edm;

  Handle<std::vector<CaloParticle>> caloParticlesHandle;
  iEvent.getByToken(caloParticleToken_, caloParticlesHandle);
  const auto &caloParticles = *caloParticlesHandle;

  Handle<std::vector<SimCluster>> simClustersHandle;
  iEvent.getByToken(simClusterToken_, simClustersHandle);
  Handle<std::unordered_map<DetId, const unsigned int>> hitMap;
  iEvent.getByToken(hitMapToken_, hitMap);

  MultiVectorManager<HGCRecHit> rechitManager;
  for (const auto &token : hitsTokens_) {
    Handle<HGCRecHitCollection> hitsHandle;
    iEvent.getByToken(token, hitsHandle);
    rechitManager.addVector(*hitsHandle);
  }

  // Create association maps
  auto hitToSimClusterMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(rechitManager.size());
  auto hitToCaloParticleMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(rechitManager.size());

  // Loop over caloParticles
  for (unsigned int cpId = 0; cpId < caloParticles.size(); ++cpId) {
    const auto &caloParticle = caloParticles[cpId];
    // Loop over simClusters in caloParticle
    for (const auto &simCluster : caloParticle.simClusters()) {
      // Loop over hits in simCluster
      for (const auto &hitAndFraction : simCluster->hits_and_fractions()) {
        auto hitMapIter = hitMap->find(hitAndFraction.first);
        if (hitMapIter != hitMap->end()) {
          unsigned int rechitIndex = hitMapIter->second;
          float fraction = hitAndFraction.second;
          hitToSimClusterMap->insert(rechitIndex, simCluster.key(), fraction);
          hitToCaloParticleMap->insert(rechitIndex, cpId, fraction);
        }
      }
    }
  }
  iEvent.put(std::move(hitToSimClusterMap), "hitToSimClusterMap");
  iEvent.put(std::move(hitToCaloParticleMap), "hitToCaloParticleMap");
}

void HitToSimClusterCaloParticleAssociatorProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));

  desc.add<edm::InputTag>("hitMap", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
  desc.add<std::vector<edm::InputTag>>("hits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  descriptions.add("hitToSimClusterCaloParticleAssociator", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(HitToSimClusterCaloParticleAssociatorProducer);
