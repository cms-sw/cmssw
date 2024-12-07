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
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

class AllTracksterToSimTracksterAssociatorsByHitsProducer : public edm::global::EDProducer<> {
public:
  explicit AllTracksterToSimTracksterAssociatorsByHitsProducer(const edm::ParameterSet&);
  ~AllTracksterToSimTracksterAssociatorsByHitsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<ticl::Trackster>>>> tracksterCollectionTokens_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<ticl::Trackster>>>> simTracksterCollectionTokens_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>>>>
      hitToTracksterMapTokens_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>>>>
      tracksterToHitMapTokens_;

  std::vector<std::pair<std::string, edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>>>>
      hitToSimTracksterMapTokens_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>>>>
      simTracksterToHitMapTokens_;

  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hitsTokens_;
  edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimClusterMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToCaloParticleMapToken_;
};

AllTracksterToSimTracksterAssociatorsByHitsProducer::AllTracksterToSimTracksterAssociatorsByHitsProducer(
    const edm::ParameterSet& pset)
    : caloParticleToken_(consumes<std::vector<CaloParticle>>(pset.getParameter<edm::InputTag>("caloParticles"))),
      hitToSimClusterMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToSimClusterMap"))),
      hitToCaloParticleMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToCaloParticleMap"))) {
  const auto& tracksterCollections = pset.getParameter<std::vector<edm::InputTag>>("tracksterCollections");
  for (const auto& tag : tracksterCollections) {
    std::string label = tag.label();
    if (tag.instance() != "") {
      label += tag.instance();
    }
    tracksterCollectionTokens_.emplace_back(label, consumes<std::vector<ticl::Trackster>>(tag));
    hitToTracksterMapTokens_.emplace_back(label,
                                          consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
                                              edm::InputTag("allHitToTracksterAssociations", "hitTo" + label)));
    tracksterToHitMapTokens_.emplace_back(label,
                                          consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
                                              edm::InputTag("allHitToTracksterAssociations", label + "ToHit")));
  }

  const auto& simTracksterCollections = pset.getParameter<std::vector<edm::InputTag>>("simTracksterCollections");
  for (const auto& tag : simTracksterCollections) {
    std::string label = tag.label();
    if (tag.instance() != "") {
      label += tag.instance();
    }
    simTracksterCollectionTokens_.emplace_back(label, consumes<std::vector<ticl::Trackster>>(tag));
    hitToSimTracksterMapTokens_.emplace_back(label,
                                             consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
                                                 edm::InputTag("allHitToTracksterAssociations", "hitTo" + label)));
    simTracksterToHitMapTokens_.emplace_back(label,
                                             consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
                                                 edm::InputTag("allHitToTracksterAssociations", label + "ToHit")));
  }

  // Hits
  auto hitsTags = pset.getParameter<std::vector<edm::InputTag>>("hits");
  for (const auto& tag : hitsTags) {
    hitsTokens_.push_back(consumes<HGCRecHitCollection>(tag));
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

void AllTracksterToSimTracksterAssociatorsByHitsProducer::produce(edm::StreamID,
                                                                  edm::Event& iEvent,
                                                                  const edm::EventSetup&) const {
  using namespace edm;

  MultiVectorManager<HGCRecHit> rechitManager;
  for (const auto& token : hitsTokens_) {
    Handle<HGCRecHitCollection> hitsHandle;
    iEvent.getByToken(token, hitsHandle);
    rechitManager.addVector(*hitsHandle);
  }

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimClusterMapHandle;
  iEvent.getByToken(hitToSimClusterMapToken_, hitToSimClusterMapHandle);
  const auto& hitToSimClusterMap = *hitToSimClusterMapHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToCaloParticleMapHandle;
  iEvent.getByToken(hitToCaloParticleMapToken_, hitToCaloParticleMapHandle);
  const auto& hitToCaloParticleMap = *hitToCaloParticleMapHandle;

  Handle<std::vector<CaloParticle>> caloParticlesHandle;
  iEvent.getByToken(caloParticleToken_, caloParticlesHandle);

  for (const auto& tracksterToken : tracksterCollectionTokens_) {
    Handle<std::vector<ticl::Trackster>> recoTrackstersHandle;
    iEvent.getByToken(tracksterToken.second, recoTrackstersHandle);
    const auto& recoTracksters = *recoTrackstersHandle;

    // Retrieve the correct HitToTracksterMap for the current trackster collection
    Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToTracksterMapHandle;
    auto tracksterMapTokenIter = std::find_if(
        hitToTracksterMapTokens_.begin(), hitToTracksterMapTokens_.end(), [&tracksterToken](const auto& pair) {
          return pair.first == tracksterToken.first;
        });
    if (tracksterMapTokenIter != hitToTracksterMapTokens_.end()) {
      iEvent.getByToken(tracksterMapTokenIter->second, hitToTracksterMapHandle);
    }
    const auto& hitToTracksterMap = *hitToTracksterMapHandle;

    // Retrieve the correct TracksterToHitMap for the current trackster collection
    Handle<ticl::AssociationMap<ticl::mapWithFraction>> tracksterToHitMapHandle;
    auto tracksterToHitMapTokenIter = std::find_if(
        tracksterToHitMapTokens_.begin(), tracksterToHitMapTokens_.end(), [&tracksterToken](const auto& pair) {
          return pair.first == tracksterToken.first;
        });
    if (tracksterToHitMapTokenIter != tracksterToHitMapTokens_.end()) {
      iEvent.getByToken(tracksterToHitMapTokenIter->second, tracksterToHitMapHandle);
    }
    const auto& tracksterToHitMap = *tracksterToHitMapHandle;

    for (const auto& simTracksterToken : simTracksterCollectionTokens_) {
      Handle<std::vector<ticl::Trackster>> simTrackstersHandle;
      iEvent.getByToken(simTracksterToken.second, simTrackstersHandle);
      const auto& simTracksters = *simTrackstersHandle;

      // Retrieve the correct HitToSimTracksterMap for the current simTrackster collection
      Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimTracksterMapHandle;
      auto simTracksterMapTokenIter =
          std::find_if(hitToSimTracksterMapTokens_.begin(),
                       hitToSimTracksterMapTokens_.end(),
                       [&simTracksterToken](const auto& pair) { return pair.first == simTracksterToken.first; });
      if (simTracksterMapTokenIter != hitToSimTracksterMapTokens_.end()) {
        iEvent.getByToken(simTracksterMapTokenIter->second, hitToSimTracksterMapHandle);
      }
      const auto& hitToSimTracksterMap = *hitToSimTracksterMapHandle;

      // Retrieve the correct SimTracksterToHitMap for the current simTrackster collection
      Handle<ticl::AssociationMap<ticl::mapWithFraction>> simTracksterToHitMapHandle;
      auto simTracksterToHitMapTokenIter =
          std::find_if(simTracksterToHitMapTokens_.begin(),
                       simTracksterToHitMapTokens_.end(),
                       [&simTracksterToken](const auto& pair) { return pair.first == simTracksterToken.first; });
      if (simTracksterToHitMapTokenIter != simTracksterToHitMapTokens_.end()) {
        iEvent.getByToken(simTracksterToHitMapTokenIter->second, simTracksterToHitMapHandle);
      }
      const auto& simTracksterToHitMap = *simTracksterToHitMapHandle;

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
        edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, tracksterIndex);
        float recoToSimScoresDenominator = 0.f;
        const auto& recoTracksterHitsAndFractions = tracksterToHitMap[tracksterIndex];
        ticl::AssociationMap<ticl::mapWithFraction> hitToAssociatedSimTracksterMap(
            recoTracksterHitsAndFractions.size());
        std::vector<unsigned int> associatedSimTracksterIndices;
        for (unsigned int i = 0; i < recoTracksterHitsAndFractions.size(); ++i) {
          const auto& hitElement = recoTracksterHitsAndFractions[i];
          unsigned int hitIndex = hitElement.index();
          float recoFraction = hitElement.fraction();
          const auto& recHit = rechitManager[hitIndex];
          float squaredRecoFraction = recoFraction * recoFraction;
          float rechitEnergy = recHit.energy();
          float squaredRecHitEnergy = rechitEnergy * rechitEnergy;
          recoToSimScoresDenominator += squaredRecoFraction * squaredRecHitEnergy;

          const auto& hitToSimTracksterVec = hitToSimTracksterMap[hitIndex];
          for (const auto& simTracksterElement : hitToSimTracksterVec) {
            auto simTracksterIndex = simTracksterElement.index();
            const auto& simTrackster = simTracksters[simTracksterIndex];
            auto& seed = simTrackster.seedID();
            float simFraction = 0;
            if (seed == caloParticlesHandle.id()) {
              unsigned int caloParticleIndex = simTrackster.seedIndex();
              auto it =
                  std::find_if(hitToCaloParticleMap[hitIndex].begin(),
                               hitToCaloParticleMap[hitIndex].end(),
                               [caloParticleIndex](const auto& pair) { return pair.index() == caloParticleIndex; });
              if (it != hitToCaloParticleMap[hitIndex].end()) {
                simFraction = it->fraction();
              }
            } else {
              unsigned int simClusterIndex = simTracksters[simTracksterIndex].seedIndex();
              auto it = std::find_if(hitToSimClusterMap[hitIndex].begin(),
                                     hitToSimClusterMap[hitIndex].end(),
                                     [simClusterIndex](const auto& pair) { return pair.index() == simClusterIndex; });
              if (it != hitToSimClusterMap[hitIndex].end()) {
                simFraction = it->fraction();
              }
            }
            hitToAssociatedSimTracksterMap.insert(i, simTracksterIndex, simFraction);
            associatedSimTracksterIndices.push_back(simTracksterIndex);
          }
        }
        std::sort(associatedSimTracksterIndices.begin(), associatedSimTracksterIndices.end());
        associatedSimTracksterIndices.erase(
            std::unique(associatedSimTracksterIndices.begin(), associatedSimTracksterIndices.end()),
            associatedSimTracksterIndices.end());

        // Add missing sim tracksters with 0 shared energy to hitToAssociatedSimTracksterMap
        for (unsigned int i = 0; i < recoTracksterHitsAndFractions.size(); ++i) {
          unsigned int hitId = recoTracksterHitsAndFractions[i].index();
          const auto& simTracksterVec = hitToSimTracksterMap[hitId];
          for (unsigned int simTracksterIndex : associatedSimTracksterIndices) {
            if (std::find_if(simTracksterVec.begin(), simTracksterVec.end(), [simTracksterIndex](const auto& pair) {
                  return pair.index() == simTracksterIndex;
                }) == simTracksterVec.end()) {
              hitToAssociatedSimTracksterMap.insert(i, simTracksterIndex, 0);
            }
          }
        }

        const float invDenominator = 1.f / recoToSimScoresDenominator;

        for (unsigned int i = 0; i < recoTracksterHitsAndFractions.size(); ++i) {
          unsigned int hitIndex = recoTracksterHitsAndFractions[i].index();
          const auto& recHit = rechitManager[hitIndex];
          float recoFraction = recoTracksterHitsAndFractions[i].fraction();
          float squaredRecoFraction = recoFraction * recoFraction;
          float squaredRecHitEnergy = recHit.energy() * recHit.energy();
          float recoSharedEnergy = recHit.energy() * recoFraction;
          const auto& simTracksterVec = hitToAssociatedSimTracksterMap[i];
          for (const auto& simTracksterElement : simTracksterVec) {
            auto simTracksterIndex = simTracksterElement.index();
            auto simFraction = simTracksterElement.fraction();
            edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersHandle, simTracksterIndex);
            float sharedEnergy = std::min(simFraction * recHit.energy(), recoSharedEnergy);
            float squaredFraction =
                std::min(squaredRecoFraction, (recoFraction - simFraction) * (recoFraction - simFraction));
            float score = invDenominator * squaredFraction * squaredRecHitEnergy;
            tracksterToSimTracksterMap->insert(recoTracksterRef, simTracksterRef, sharedEnergy, score);
          }
        }
      }

      // Reverse mapping: SimTrackster -> RecoTrackster
      for (unsigned int tracksterIndex = 0; tracksterIndex < simTracksters.size(); ++tracksterIndex) {
        edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersHandle, tracksterIndex);
        float simToRecoScoresDenominator = 0.f;
        const auto& simTracksterHitsAndFractions = simTracksterToHitMap[tracksterIndex];
        ticl::AssociationMap<ticl::mapWithFraction> hitToAssociatedRecoTracksterMap(
            simTracksterHitsAndFractions.size());
        std::vector<unsigned int> associatedRecoTracksterIndices;
        const auto& simTrackster = simTracksters[tracksterIndex];
        auto& seed = simTrackster.seedID();
        unsigned int simObjectIndex = simTrackster.seedIndex();
        bool isSimTracksterFromCP = (seed == caloParticlesHandle.id());
        std::vector<float> simFractions(simTracksterHitsAndFractions.size(), 0.f);
        for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
          auto hitIndex = simTracksterHitsAndFractions[i].index();
          auto it = isSimTracksterFromCP
                        ? (std::find_if(hitToCaloParticleMap[hitIndex].begin(),
                                        hitToCaloParticleMap[hitIndex].end(),
                                        [simObjectIndex](const auto& pair) { return pair.index() == simObjectIndex; }))
                        : std::find_if(hitToSimClusterMap[hitIndex].begin(),
                                       hitToSimClusterMap[hitIndex].end(),
                                       [simObjectIndex](const auto& pair) { return pair.index() == simObjectIndex; });
          if (it != hitToCaloParticleMap[hitIndex].end() and it != hitToSimClusterMap[hitIndex].end()) {
            simFractions[i] = it->fraction();
          }
          float simFraction = simFractions[i];
          const auto& recHit = rechitManager[hitIndex];
          float squaredSimFraction = simFraction * simFraction;
          float squaredRecHitEnergy = recHit.energy() * recHit.energy();
          simToRecoScoresDenominator += squaredSimFraction * squaredRecHitEnergy;

          const auto& hitToRecoTracksterVec = hitToTracksterMap[hitIndex];
          for (const auto& recoTracksterElement : hitToRecoTracksterVec) {
            unsigned int recoTracksterIndex = recoTracksterElement.index();
            float recoFraction = recoTracksterElement.fraction();
            hitToAssociatedRecoTracksterMap.insert(i, recoTracksterIndex, recoFraction);
            associatedRecoTracksterIndices.push_back(recoTracksterIndex);
          }
        }

        std::sort(associatedRecoTracksterIndices.begin(), associatedRecoTracksterIndices.end());
        associatedRecoTracksterIndices.erase(
            std::unique(associatedRecoTracksterIndices.begin(), associatedRecoTracksterIndices.end()),
            associatedRecoTracksterIndices.end());

        for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
          unsigned int hitIndex = simTracksterHitsAndFractions[i].index();
          const auto& hitToRecoTracksterVec = hitToTracksterMap[hitIndex];
          for (unsigned int recoTracksterIndex : associatedRecoTracksterIndices) {
            if (std::find_if(
                    hitToRecoTracksterVec.begin(), hitToRecoTracksterVec.end(), [recoTracksterIndex](const auto& pair) {
                      return pair.index() == recoTracksterIndex;
                    }) == hitToRecoTracksterVec.end()) {
              hitToAssociatedRecoTracksterMap.insert(i, recoTracksterIndex, 0);
            }
          }
        }

        const float invDenominator = 1.f / simToRecoScoresDenominator;

        for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
          const auto& hitIndex = simTracksterHitsAndFractions[i].index();
          float simFraction = simFractions[i];
          const auto& recHit = rechitManager[hitIndex];
          float squaredSimFraction = simFraction * simFraction;
          float squaredRecHitEnergy = recHit.energy() * recHit.energy();
          float simSharedEnergy = recHit.energy() * simFraction;

          const auto& hitToRecoTracksterVec = hitToAssociatedRecoTracksterMap[i];
          for (const auto& recoTracksterElement : hitToRecoTracksterVec) {
            auto recoTracksterIndex = recoTracksterElement.index();
            float recoFraction = recoTracksterElement.fraction();
            edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, recoTracksterIndex);
            float sharedEnergy = std::min(recoFraction * recHit.energy(), simSharedEnergy);
            float squaredFraction =
                std::min(squaredSimFraction, (recoFraction - simFraction) * (recoFraction - simFraction));
            float score = invDenominator * squaredFraction * squaredRecHitEnergy;
            simTracksterToTracksterMap->insert(simTracksterRef, recoTracksterRef, sharedEnergy, score);
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

void AllTracksterToSimTracksterAssociatorsByHitsProducer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>(
      "tracksterCollections", {edm::InputTag("ticlTrackstersCLUE3DHigh"), edm::InputTag("ticlTrackstersLinks")});
  desc.add<std::vector<edm::InputTag>>(
      "simTracksterCollections", {edm::InputTag("ticlSimTracksters"), edm::InputTag("ticlSimTracksters", "fromCPs")});
  desc.add<std::vector<edm::InputTag>>("hits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  desc.add<edm::InputTag>("hitToSimClusterMap",
                          edm::InputTag("hitToSimClusterCaloParticleAssociator", "hitToSimClusterMap"));
  desc.add<edm::InputTag>("hitToCaloParticleMap",
                          edm::InputTag("hitToSimClusterCaloParticleAssociator", "hitToCaloParticleMap"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));

  descriptions.add("AllTracksterToSimTracksterAssociatorsByHitsProducer", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(AllTracksterToSimTracksterAssociatorsByHitsProducer);
