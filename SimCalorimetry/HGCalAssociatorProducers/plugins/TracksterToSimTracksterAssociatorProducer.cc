// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024

#include "TracksterToSimTracksterAssociatorProducer.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

TracksterToSimTracksterAssociatorProducer::TracksterToSimTracksterAssociatorProducer(const edm::ParameterSet& pset)
    : recoTracksterCollectionToken_(
          consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("tracksters"))),
      simTracksterCollectionToken_(
          consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("simTracksters"))),
      layerClustersCollectionToken_(
          consumes<std::vector<reco::CaloCluster>>(pset.getParameter<edm::InputTag>("layerClusters"))),
      LayerClusterToTracksterMapToken_(
          consumes<
              ticl::AssociationMap<ticl::mapWithFraction, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>(
              pset.getParameter<edm::InputTag>("tracksterMap"))),
      LayerClusterToSimTracksterMapToken_(
          consumes<
              ticl::AssociationMap<ticl::mapWithFraction, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>(
              pset.getParameter<edm::InputTag>("simTracksterMap"))) {
  produces<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      "tracksterToSimTracksterMap");
  produces<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      "simTracksterToTracksterMap");
}

TracksterToSimTracksterAssociatorProducer::~TracksterToSimTracksterAssociatorProducer() {}

void TracksterToSimTracksterAssociatorProducer::produce(edm::StreamID,
                                                        edm::Event& iEvent,
                                                        const edm::EventSetup& iSetup) const {
  edm::Handle<std::vector<ticl::Trackster>> recoTrackstersHandle;
  iEvent.getByToken(recoTracksterCollectionToken_, recoTrackstersHandle);
  const auto& recoTracksters = *recoTrackstersHandle;

  edm::Handle<std::vector<ticl::Trackster>> simTrackstersHandle;
  iEvent.getByToken(simTracksterCollectionToken_, simTrackstersHandle);
  const auto& simTracksters = *simTrackstersHandle;

  edm::Handle<std::vector<reco::CaloCluster>> layerClustersHandle;
  iEvent.getByToken(layerClustersCollectionToken_, layerClustersHandle);
  const auto& layerClusters = *layerClustersHandle;

  edm::Handle<ticl::AssociationMap<ticl::mapWithFraction, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>
      layerClusterToTracksterMapHandle;
  iEvent.getByToken(LayerClusterToTracksterMapToken_, layerClusterToTracksterMapHandle);
  const auto& layerClusterToTracksterMap = *layerClusterToTracksterMapHandle;

  edm::Handle<ticl::AssociationMap<ticl::mapWithFraction, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>
      layerClusterToSimTracksterMapHandle;
  iEvent.getByToken(LayerClusterToSimTracksterMapToken_, layerClusterToSimTracksterMapHandle);
  const auto& layerClusterToSimTracksterMap = *layerClusterToSimTracksterMapHandle;

  auto tracksterToSimTracksterMap = std::make_unique<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      recoTrackstersHandle, simTrackstersHandle, iEvent);
  auto simTracksterToTracksterMap = std::make_unique<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      simTrackstersHandle, recoTrackstersHandle, iEvent);

  for (unsigned int tracksterIndex = 0; tracksterIndex < recoTracksters.size(); ++tracksterIndex) {
    const auto& recoTrackster = recoTracksters[tracksterIndex];
    edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, tracksterIndex);
    const auto& layerClustersIds = recoTrackster.vertices();
    float recoToSimScoresDenominator = 0.f;
    ticl::mapWithFraction layerClusterToAssociatedSimTracksterMap(layerClustersIds.size());
    std::vector<unsigned int> associatedSimTracksterIndices;
    for (unsigned int i = 0; i < layerClustersIds.size(); ++i) {
      unsigned int layerClusterId = layerClustersIds[i];
      const auto& layerCluster = layerClusters[layerClusterId];
      float recoFraction = 1.f / recoTrackster.vertex_multiplicity(i);
      float squaredRecoFraction = recoFraction * recoFraction;
      float squaredLayerClusterEnergy = layerCluster.energy() * layerCluster.energy();
      recoToSimScoresDenominator += squaredLayerClusterEnergy * squaredRecoFraction;
      const auto& simTracksterVec = layerClusterToSimTracksterMap[layerClusterId];
      for (const auto& [simTracksterIndex, simSharedEnergy] : simTracksterVec) {
        layerClusterToAssociatedSimTracksterMap[i].push_back({simTracksterIndex, simSharedEnergy});
        associatedSimTracksterIndices.push_back(simTracksterIndex);
      }
    }

    // Keep only unique associatedSimTracksterIndices
    std::sort(associatedSimTracksterIndices.begin(), associatedSimTracksterIndices.end());
    associatedSimTracksterIndices.erase(
        std::unique(associatedSimTracksterIndices.begin(), associatedSimTracksterIndices.end()),
        associatedSimTracksterIndices.end());

    // Add missing sim tracksters with 0 shared energy to layerClusterToAssociatedSimTracksterMap
    for (unsigned int i = 0; i < layerClustersIds.size(); ++i) {
      unsigned int layerClusterId = layerClustersIds[i];
      const auto& simTracksterVec = layerClusterToSimTracksterMap[layerClusterId];
      for (unsigned int simTracksterIndex : associatedSimTracksterIndices) {
        if (std::find_if(simTracksterVec.begin(), simTracksterVec.end(), [simTracksterIndex](const auto& pair) {
              return pair.first == simTracksterIndex;
            }) == simTracksterVec.end()) {
          layerClusterToAssociatedSimTracksterMap[i].push_back({simTracksterIndex, 0.f});
        }
      }
    }

    const float invDenominator = 1.f / recoToSimScoresDenominator;

    for (unsigned int i = 0; i < layerClustersIds.size(); ++i) {
      unsigned int layerClusterId = layerClustersIds[i];
      const auto& layerCluster = layerClusters[layerClusterId];
      float recoFraction = 1.f / recoTrackster.vertex_multiplicity(i);
      float squaredRecoFraction = recoFraction * recoFraction;
      float squaredLayerClusterEnergy = layerCluster.energy() * layerCluster.energy();
      float recoSharedEnergy = layerCluster.energy() * recoFraction;
      float invLayerClusterEnergy = 1.f / layerCluster.energy();
      const auto& simTracksterVec = layerClusterToAssociatedSimTracksterMap[i];
      for (const auto& [simTracksterIndex, simSharedEnergy] : simTracksterVec) {
        edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersHandle, simTracksterIndex);
        float sharedEnergy = std::min(simSharedEnergy, recoSharedEnergy);
        float simFraction = simSharedEnergy * invLayerClusterEnergy;
        float score = invDenominator *
                      std::min(squaredRecoFraction, (recoFraction - simFraction) * (recoFraction - simFraction)) *
                      squaredLayerClusterEnergy;
        tracksterToSimTracksterMap->insert(recoTracksterRef, simTracksterRef, sharedEnergy, score);
      }
    }
  }

  for (unsigned int tracksterIndex = 0; tracksterIndex < simTracksters.size(); ++tracksterIndex) {
    const auto& simTrackster = simTracksters[tracksterIndex];
    edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersHandle, tracksterIndex);
    const auto& layerClustersIds = simTrackster.vertices();
    float simToRecoScoresDenominator = 0.f;
    ticl::mapWithFraction layerClusterToAssociatedTracksterMap(layerClustersIds.size());
    std::vector<unsigned int> associatedRecoTracksterIndices;
    for (unsigned int i = 0; i < layerClustersIds.size(); ++i) {
      unsigned int layerClusterId = layerClustersIds[i];
      const auto& layerCluster = layerClusters[layerClusterId];
      float simFraction = 1.f / simTrackster.vertex_multiplicity(i);
      float squaredSimFraction = simFraction * simFraction;
      float squaredLayerClusterEnergy = layerCluster.energy() * layerCluster.energy();
      simToRecoScoresDenominator += squaredLayerClusterEnergy * squaredSimFraction;
      const auto& recoTracksterVec = layerClusterToTracksterMap[layerClusterId];
      for (const auto& [recoTracksterIndex, recoSharedEnergy] : recoTracksterVec) {
        layerClusterToAssociatedTracksterMap[i].push_back({recoTracksterIndex, recoSharedEnergy});
        associatedRecoTracksterIndices.push_back(recoTracksterIndex);
      }
    }
    // keep only unique associatedRecoTracksterIndices
    std::sort(associatedRecoTracksterIndices.begin(), associatedRecoTracksterIndices.end());
    associatedRecoTracksterIndices.erase(
        std::unique(associatedRecoTracksterIndices.begin(), associatedRecoTracksterIndices.end()),
        associatedRecoTracksterIndices.end());
    // for each layer cluster, loop over associatedRecoTracksterIndices and add the missing reco tracksters with 0 shared energy to layerClusterToAssociatedTracksterMap
    for (unsigned int i = 0; i < layerClustersIds.size(); ++i) {
      unsigned int layerClusterId = layerClustersIds[i];
      const auto& recoTracksterVec = layerClusterToTracksterMap[layerClusterId];
      for (unsigned int recoTracksterIndex : associatedRecoTracksterIndices) {
        if (std::find_if(recoTracksterVec.begin(), recoTracksterVec.end(), [recoTracksterIndex](const auto& pair) {
              return pair.first == recoTracksterIndex;
            }) == recoTracksterVec.end()) {
          layerClusterToAssociatedTracksterMap[i].push_back({recoTracksterIndex, 0.f});
        }
      }
    }

    const float invDenominator = 1.f / simToRecoScoresDenominator;

    for (unsigned int i = 0; i < layerClustersIds.size(); ++i) {
      unsigned int layerClusterId = layerClustersIds[i];
      const auto& layerCluster = layerClusters[layerClusterId];
      float simFraction = 1.f / simTrackster.vertex_multiplicity(i);
      float squaredSimFraction = simFraction * simFraction;
      float squaredLayerClusterEnergy = layerCluster.energy() * layerCluster.energy();
      float simSharedEnergy = layerCluster.energy() * simFraction;
      float invLayerClusterEnergy = 1.f / layerCluster.energy();
      const auto& recoTracksterVec = layerClusterToAssociatedTracksterMap[i];
      for (const auto& [recoTracksterIndex, recoSharedEnergy] : recoTracksterVec) {
        edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, recoTracksterIndex);
        float sharedEnergy = std::min(recoSharedEnergy, simSharedEnergy);
        float recoFraction = recoSharedEnergy * invLayerClusterEnergy;
        float score = invDenominator *
                      std::min(squaredSimFraction, (simFraction - recoFraction) * (simFraction - recoFraction)) *
                      squaredLayerClusterEnergy;
        simTracksterToTracksterMap->insert(tracksterIndex, recoTracksterIndex, sharedEnergy, score);
      }
    }
  }
  tracksterToSimTracksterMap->sort(true);
  simTracksterToTracksterMap->sort(true);
  iEvent.put(std::move(tracksterToSimTracksterMap), "tracksterToSimTracksterMap");
  iEvent.put(std::move(simTracksterToTracksterMap), "simTracksterToTracksterMap");
}

void TracksterToSimTracksterAssociatorProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksters", edm::InputTag("trackstersProducer"));
  desc.add<edm::InputTag>("simTracksters", edm::InputTag("simTrackstersProducer"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("tracksterMap", edm::InputTag("tracksterAssociatorProducer"));
  desc.add<edm::InputTag>("simTracksterMap", edm::InputTag("simTracksterAssociatorProducer"));
  descriptions.add("tracksterToSimTracksterAssociatorProducer", desc);
}

DEFINE_FWK_MODULE(TracksterToSimTracksterAssociatorProducer);
