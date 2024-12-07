// Author: Felice Pantaleo, felice.pantaleo@cern.ch 08/2024

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

class AllTracksterToSimTracksterAssociatorsByLCsProducer : public edm::global::EDProducer<> {
public:
  explicit AllTracksterToSimTracksterAssociatorsByLCsProducer(const edm::ParameterSet&);
  ~AllTracksterToSimTracksterAssociatorsByLCsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<ticl::Trackster>>>> tracksterCollectionTokens_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<ticl::Trackster>>>> simTracksterCollectionTokens_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  std::vector<std::pair<
      std::string,
      edm::EDGetTokenT<
          ticl::AssociationMap<ticl::mapWithSharedEnergy, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>>>
      layerClusterToTracksterMapTokens_;
  std::vector<std::pair<
      std::string,
      edm::EDGetTokenT<
          ticl::AssociationMap<ticl::mapWithSharedEnergy, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>>>
      layerClusterToSimTracksterMapTokens_;
};

AllTracksterToSimTracksterAssociatorsByLCsProducer::AllTracksterToSimTracksterAssociatorsByLCsProducer(
    const edm::ParameterSet& pset)
    : layerClustersToken_(consumes<std::vector<reco::CaloCluster>>(pset.getParameter<edm::InputTag>("layerClusters"))) {
  const auto& tracksterCollections = pset.getParameter<std::vector<edm::InputTag>>("tracksterCollections");
  for (const auto& tag : tracksterCollections) {
    std::string label = tag.label();
    if (tag.instance() != "") {
      label += tag.instance();
    }
    tracksterCollectionTokens_.emplace_back(label, consumes<std::vector<ticl::Trackster>>(tag));
    layerClusterToTracksterMapTokens_.emplace_back(
        label,
        consumes<
            ticl::AssociationMap<ticl::mapWithSharedEnergy, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>(
            edm::InputTag("allLayerClusterToTracksterAssociations", label)));
  }

  const auto& simTracksterCollections = pset.getParameter<std::vector<edm::InputTag>>("simTracksterCollections");
  for (const auto& tag : simTracksterCollections) {
    std::string label = tag.label();
    if (tag.instance() != "") {
      label += tag.instance();
    }
    simTracksterCollectionTokens_.emplace_back(label, consumes<std::vector<ticl::Trackster>>(tag));
    layerClusterToSimTracksterMapTokens_.emplace_back(
        label,
        consumes<
            ticl::AssociationMap<ticl::mapWithSharedEnergy, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>(
            edm::InputTag("allLayerClusterToTracksterAssociations", label)));
  }

  // Produce separate association maps for each trackster-simTrackster combination
  for (const auto& tracksterToken : tracksterCollectionTokens_) {
    for (const auto& simTracksterToken : simTracksterCollectionTokens_) {
      std::string instanceLabel = tracksterToken.first + "To" + simTracksterToken.first;
      produces<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                    std::vector<ticl::Trackster>,
                                    std::vector<ticl::Trackster>>>(instanceLabel);
      std::string reverseInstanceLabel = simTracksterToken.first + "To" + tracksterToken.first;
      produces<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                    std::vector<ticl::Trackster>,
                                    std::vector<ticl::Trackster>>>(reverseInstanceLabel);
    }
  }
}

void AllTracksterToSimTracksterAssociatorsByLCsProducer::produce(edm::StreamID,
                                                                 edm::Event& iEvent,
                                                                 const edm::EventSetup&) const {
  using namespace edm;

  Handle<std::vector<reco::CaloCluster>> layerClustersHandle;
  iEvent.getByToken(layerClustersToken_, layerClustersHandle);
  const auto& layerClusters = *layerClustersHandle;

  for (const auto& tracksterToken : tracksterCollectionTokens_) {
    Handle<std::vector<ticl::Trackster>> recoTrackstersHandle;
    iEvent.getByToken(tracksterToken.second, recoTrackstersHandle);
    const auto& recoTracksters = *recoTrackstersHandle;

    // Retrieve the correct LayerClusterToTracksterMap for the current trackster collection
    Handle<ticl::AssociationMap<ticl::mapWithSharedEnergy, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>
        layerClusterToTracksterMapHandle;
    auto tracksterMapTokenIter =
        std::find_if(layerClusterToTracksterMapTokens_.begin(),
                     layerClusterToTracksterMapTokens_.end(),
                     [&tracksterToken](const auto& pair) { return pair.first == tracksterToken.first; });
    if (tracksterMapTokenIter != layerClusterToTracksterMapTokens_.end()) {
      iEvent.getByToken(tracksterMapTokenIter->second, layerClusterToTracksterMapHandle);
    }
    const auto& layerClusterToTracksterMap = *layerClusterToTracksterMapHandle;

    for (const auto& simTracksterToken : simTracksterCollectionTokens_) {
      Handle<std::vector<ticl::Trackster>> simTrackstersHandle;
      iEvent.getByToken(simTracksterToken.second, simTrackstersHandle);
      const auto& simTracksters = *simTrackstersHandle;

      // Retrieve the correct LayerClusterToSimTracksterMap for the current simTrackster collection
      Handle<
          ticl::AssociationMap<ticl::mapWithSharedEnergy, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>
          layerClusterToSimTracksterMapHandle;
      auto simTracksterMapTokenIter =
          std::find_if(layerClusterToSimTracksterMapTokens_.begin(),
                       layerClusterToSimTracksterMapTokens_.end(),
                       [&simTracksterToken](const auto& pair) { return pair.first == simTracksterToken.first; });
      if (simTracksterMapTokenIter != layerClusterToSimTracksterMapTokens_.end()) {
        iEvent.getByToken(simTracksterMapTokenIter->second, layerClusterToSimTracksterMapHandle);
      }
      const auto& layerClusterToSimTracksterMap = *layerClusterToSimTracksterMapHandle;

      // Create the association maps
      auto tracksterToSimTracksterMap = std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                                              std::vector<ticl::Trackster>,
                                                                              std::vector<ticl::Trackster>>>(
          recoTrackstersHandle, simTrackstersHandle, iEvent);
      auto simTracksterToTracksterMap = std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                                              std::vector<ticl::Trackster>,
                                                                              std::vector<ticl::Trackster>>>(
          simTrackstersHandle, recoTrackstersHandle, iEvent);

      for (unsigned int tracksterIndex = 0; tracksterIndex < recoTracksters.size(); ++tracksterIndex) {
        const auto& recoTrackster = recoTracksters[tracksterIndex];
        edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, tracksterIndex);
        const auto& layerClustersIds = recoTrackster.vertices();
        float recoToSimScoresDenominator = 0.f;
        ticl::AssociationMap<ticl::mapWithSharedEnergy> layerClusterToAssociatedSimTracksterMap(
            layerClustersIds.size());
        std::vector<unsigned int> associatedSimTracksterIndices;
        for (unsigned int i = 0; i < layerClustersIds.size(); ++i) {
          unsigned int layerClusterId = layerClustersIds[i];
          const auto& layerCluster = layerClusters[layerClusterId];
          float recoFraction = 1.f / recoTrackster.vertex_multiplicity(i);
          float squaredRecoFraction = recoFraction * recoFraction;
          float squaredLayerClusterEnergy = layerCluster.energy() * layerCluster.energy();
          recoToSimScoresDenominator += squaredLayerClusterEnergy * squaredRecoFraction;
          const auto& simTracksterVec = layerClusterToSimTracksterMap[layerClusterId];
          for (const auto& simTracksterElement : simTracksterVec) {
            auto simTracksterIndex = simTracksterElement.index();
            auto simSharedEnergy = simTracksterElement.sharedEnergy();
            layerClusterToAssociatedSimTracksterMap[i].emplace_back(simTracksterIndex, simSharedEnergy);
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
                  return pair.index() == simTracksterIndex;
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
          for (const auto& simTracksterElement : simTracksterVec) {
            auto simTracksterIndex = simTracksterElement.index();
            float simSharedEnergy = simTracksterElement.sharedEnergy();
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
        ticl::AssociationMap<ticl::mapWithSharedEnergy> layerClusterToAssociatedTracksterMap(layerClustersIds.size());
        std::vector<unsigned int> associatedRecoTracksterIndices;
        for (unsigned int i = 0; i < layerClustersIds.size(); ++i) {
          unsigned int layerClusterId = layerClustersIds[i];
          const auto& layerCluster = layerClusters[layerClusterId];
          float simFraction = 1.f / simTrackster.vertex_multiplicity(i);
          float squaredSimFraction = simFraction * simFraction;
          float squaredLayerClusterEnergy = layerCluster.energy() * layerCluster.energy();
          simToRecoScoresDenominator += squaredLayerClusterEnergy * squaredSimFraction;
          const auto& recoTracksterVec = layerClusterToTracksterMap[layerClusterId];
          for (const auto& recoTracksterElement : recoTracksterVec) {
            auto recoTracksterIndex = recoTracksterElement.index();
            auto recoSharedEnergy = recoTracksterElement.sharedEnergy();
            layerClusterToAssociatedTracksterMap[i].emplace_back(recoTracksterIndex, recoSharedEnergy);
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
                  return pair.index() == recoTracksterIndex;
                }) == recoTracksterVec.end()) {
              layerClusterToAssociatedTracksterMap[i].emplace_back(recoTracksterIndex, 0.f);
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
          for (const auto& recoTracksterElement : recoTracksterVec) {
            auto recoTracksterIndex = recoTracksterElement.index();
            float recoSharedEnergy = recoTracksterElement.sharedEnergy();
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
      // Sort the maps by score in ascending order
      tracksterToSimTracksterMap->sort([](const auto& a, const auto& b) { return a.score() < b.score(); });
      simTracksterToTracksterMap->sort([](const auto& a, const auto& b) { return a.score() < b.score(); });

      // After populating the maps, store them in the event
      iEvent.put(std::move(tracksterToSimTracksterMap), tracksterToken.first + "To" + simTracksterToken.first);
      iEvent.put(std::move(simTracksterToTracksterMap), simTracksterToken.first + "To" + tracksterToken.first);
    }
  }
}

void AllTracksterToSimTracksterAssociatorsByLCsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>(
      "tracksterCollections", {edm::InputTag("ticlTrackstersCLUE3DHigh"), edm::InputTag("ticlTrackstersLinks")});
  desc.add<std::vector<edm::InputTag>>(
      "simTracksterCollections", {edm::InputTag("ticlSimTracksters"), edm::InputTag("ticlSimTracksters", "fromCPs")});
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  descriptions.add("AllTracksterToSimTracksterAssociatorsByLCsProducer", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(AllTracksterToSimTracksterAssociatorsByLCsProducer);
