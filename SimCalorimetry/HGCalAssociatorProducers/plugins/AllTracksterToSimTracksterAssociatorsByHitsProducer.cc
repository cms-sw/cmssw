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
#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/Common/interface/MultiSpan.h"
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
  std::vector<std::pair<std::string, edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>>>>
      hitToTracksterMapTokens_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>>>>
      tracksterToHitMapTokens_;

  std::vector<std::pair<std::string, edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>>>>
      hitToSimTracksterMapTokens_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>>>>
      simTracksterToHitMapTokens_;

  edm::EDGetTokenT<edm::RefProdVector<HGCRecHitCollection>> hitsToken_;
  edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimClusterMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToCaloParticleMapToken_;
};

AllTracksterToSimTracksterAssociatorsByHitsProducer::AllTracksterToSimTracksterAssociatorsByHitsProducer(
    const edm::ParameterSet& pset)
    : hitsToken_(consumes<edm::RefProdVector<HGCRecHitCollection>>(pset.getParameter<edm::InputTag>("hits"))),
      caloParticleToken_(consumes<std::vector<CaloParticle>>(pset.getParameter<edm::InputTag>("caloParticles"))),
      hitToSimClusterMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToSimClusterMap"))),
      hitToCaloParticleMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToCaloParticleMap"))) {
  const auto& tracksterCollections = pset.getParameter<std::vector<edm::InputTag>>("tracksterCollections");

  std::string allHitToTSAccoc = pset.getParameter<std::string>("allHitToTSAccoc");
  for (const auto& tag : tracksterCollections) {
    std::string label = tag.label();
    if (!tag.instance().empty()) {
      label += tag.instance();
    }
    tracksterCollectionTokens_.emplace_back(label, consumes<std::vector<ticl::Trackster>>(tag));
    hitToTracksterMapTokens_.emplace_back(
        label, consumes<ticl::AssociationMap<ticl::mapWithFraction>>(edm::InputTag(allHitToTSAccoc, "hitTo" + label)));
    tracksterToHitMapTokens_.emplace_back(
        label, consumes<ticl::AssociationMap<ticl::mapWithFraction>>(edm::InputTag(allHitToTSAccoc, label + "ToHit")));
  }

  const auto& simTracksterCollections = pset.getParameter<std::vector<edm::InputTag>>("simTracksterCollections");
  for (const auto& tag : simTracksterCollections) {
    std::string label = tag.label();
    if (!tag.instance().empty()) {
      label += tag.instance();
    }
    simTracksterCollectionTokens_.emplace_back(label, consumes<std::vector<ticl::Trackster>>(tag));
    hitToSimTracksterMapTokens_.emplace_back(
        label, consumes<ticl::AssociationMap<ticl::mapWithFraction>>(edm::InputTag(allHitToTSAccoc, "hitTo" + label)));
    simTracksterToHitMapTokens_.emplace_back(
        label, consumes<ticl::AssociationMap<ticl::mapWithFraction>>(edm::InputTag(allHitToTSAccoc, label + "ToHit")));
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

  if (!iEvent.getHandle(hitsToken_).isValid()) {
    edm::LogWarning("AllTracksterToSimTracksterAssociatorsByHitsProducer") << "Missing MultiHGCRecHitCollection.";
    for (const auto& tracksterToken : tracksterCollectionTokens_) {
      for (const auto& simTracksterToken : simTracksterCollectionTokens_) {
        iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                         std::vector<ticl::Trackster>,
                                                         std::vector<ticl::Trackster>>>(),
                   tracksterToken.first + "To" + simTracksterToken.first);
        iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                         std::vector<ticl::Trackster>,
                                                         std::vector<ticl::Trackster>>>(),
                   simTracksterToken.first + "To" + tracksterToken.first);
      }
    }
    return;
  }

  // Protection against missing HGCRecHitCollection
  const auto hits = iEvent.get(hitsToken_);
  for (std::size_t index = 0; const auto& hgcRecHitCollection : hits) {
    if (hgcRecHitCollection->empty()) {
      LogDebug("AllTracksterToSimTracksterAssociatorsByHitsProducer")
          << "HGCRecHitCollections #" << index << " is empty.";
    }
    index++;
  }

  edm::MultiSpan<HGCRecHit> rechitSpan(hits);
  // Check if rechitSpan is empty
  if (rechitSpan.size() == 0) {
    LogDebug("AllTracksterToSimTracksterAssociatorsByHitsProducer")
        << "Only empty HGCRecHitCollections found. Association maps will be empty.";

    for (const auto& tracksterToken : tracksterCollectionTokens_) {
      for (const auto& simTracksterToken : simTracksterCollectionTokens_) {
        iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                         std::vector<ticl::Trackster>,
                                                         std::vector<ticl::Trackster>>>(),
                   tracksterToken.first + "To" + simTracksterToken.first);
        iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                         std::vector<ticl::Trackster>,
                                                         std::vector<ticl::Trackster>>>(),
                   simTracksterToken.first + "To" + tracksterToken.first);
      }
    }
    return;
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

    if (!recoTrackstersHandle.isValid()) {
      edm::LogWarning("AllTracksterToSimTracksterAssociatorsByHitsProducer")
          << "No valid Trackster collection found. Association maps will be empty.";
      for (const auto& simTracksterToken : simTracksterCollectionTokens_) {
        Handle<std::vector<ticl::Trackster>> simTrackstersHandle;

        iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                         std::vector<ticl::Trackster>,
                                                         std::vector<ticl::Trackster>>>(),
                   tracksterToken.first + "To" + simTracksterToken.first);
        iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                         std::vector<ticl::Trackster>,
                                                         std::vector<ticl::Trackster>>>(),
                   simTracksterToken.first + "To" + tracksterToken.first);
      }
      continue;
    }

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

    if (!tracksterToHitMapHandle.isValid()) {
      edm::LogError("AllTracksterToSimTracksterAssociatorsByHitsProducer") << "tracksterToHitMapHandle is invalid";
      continue;
    }

    const auto& tracksterToHitMap = *tracksterToHitMapHandle;

    for (const auto& simTracksterToken : simTracksterCollectionTokens_) {
      Handle<std::vector<ticl::Trackster>> simTrackstersHandle;
      iEvent.getByToken(simTracksterToken.second, simTrackstersHandle);

      if (!simTrackstersHandle.isValid()) {
        iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                         std::vector<ticl::Trackster>,
                                                         std::vector<ticl::Trackster>>>(),
                   tracksterToken.first + "To" + simTracksterToken.first);
        iEvent.put(std::make_unique<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                         std::vector<ticl::Trackster>,
                                                         std::vector<ticl::Trackster>>>(),
                   simTracksterToken.first + "To" + tracksterToken.first);
        continue;
      }

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

        if (tracksterToHitMap.size() == 0)
          continue;

        ticl::AssociationMap<ticl::mapWithFraction> hitToAssociatedSimTracksterMap(
            recoTracksterHitsAndFractions.size());
        std::vector<unsigned int> associatedSimTracksterIndices;

        for (unsigned int i = 0; i < recoTracksterHitsAndFractions.size(); ++i) {
          const auto& hitElement = recoTracksterHitsAndFractions[i];
          unsigned int hitIndex = hitElement.index();
          float recoFraction = hitElement.fraction();
          const auto& recHit = rechitSpan[hitIndex];
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
          const auto& recHit = rechitSpan[hitIndex];
          float recoFraction = recoTracksterHitsAndFractions[i].fraction();
          float rechitEnergy = recHit.energy();
          float squaredRecHitEnergy = rechitEnergy * rechitEnergy;
          float recoSharedEnergy = rechitEnergy * recoFraction;
          const auto& simTracksterVec = hitToAssociatedSimTracksterMap[i];
          for (const auto& simTracksterElement : simTracksterVec) {
            auto simTracksterIndex = simTracksterElement.index();
            auto simFraction = simTracksterElement.fraction();
            edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersHandle, simTracksterIndex);
            float sharedEnergy = std::min(simFraction * rechitEnergy, recoSharedEnergy);
            /* RecoToSim score logic:
             - simFraction >= 0 && recoFraction == 0 : simtrackster contains non-reco associated elements : ignore in recoToSim association
             - simFraction == 0 && recoFraction > 0 : rechits not present in sim trackster : penalty in score by recoFraction*E
             - simFraction == 1 && recoFraction == 1 : good association
             - 1 > recoFraction > simFraction > 0 :  sim does not contain some reco energy, penalty in score by the additional part : (recoFraction-simFraction)*E
             - 1 > simFraction> recoFraction > 0 : consider as good association, as there is enough sim to cover the reco
            */
            float recoMinusSimFraction = std::max(0.f, recoFraction - simFraction);
            float score = invDenominator * recoMinusSimFraction * recoMinusSimFraction * squaredRecHitEnergy;
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
          if ((isSimTracksterFromCP and it != hitToCaloParticleMap[hitIndex].end()) or
              (!isSimTracksterFromCP and it != hitToSimClusterMap[hitIndex].end())) {
            simFractions[i] = it->fraction();
          }
          float simFraction = simFractions[i];
          const auto& recHit = rechitSpan[hitIndex];
          float rechitEnergy = recHit.energy();
          float squaredSimFraction = simFraction * simFraction;
          float squaredRecHitEnergy = rechitEnergy * rechitEnergy;
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

        assert(simToRecoScoresDenominator > 0.f);
        const float invDenominator = 1.f / simToRecoScoresDenominator;
        for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
          const auto& hitIndex = simTracksterHitsAndFractions[i].index();
          float simFraction = simFractions[i];
          const auto& recHit = rechitSpan[hitIndex];
          float rechitEnergy = recHit.energy();
          float squaredRecHitEnergy = rechitEnergy * rechitEnergy;
          float simSharedEnergy = rechitEnergy * simFraction;
          const auto& hitToRecoTracksterVec = hitToAssociatedRecoTracksterMap[i];
          for (const auto& recoTracksterElement : hitToRecoTracksterVec) {
            auto recoTracksterIndex = recoTracksterElement.index();
            float recoFraction =
                recoTracksterElement.fraction();  // Either zero or one when no sharing of rechits between tracksters
            edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, recoTracksterIndex);
            float sharedEnergy = std::min(recoFraction * rechitEnergy, simSharedEnergy);
            /* SimToReco score logic:
             - simFraction = 0 && recoFraction >= 0 : trackster contains non-sim associated elements : ignore in simToReco
             - simFraction > 0 && recoFraction == 0 : simhits not present in reco trackster : penalty in score by simFraction*E
             - simFraction == 1 && recoFraction == 1 : good association
             - 1 > simFraction > recoFraction > 0 :  we are missing some sim energy, penalty in score by the missing part : (simFraction-recoFraction)*E
             - 1 > recoFraction > simFraction > 0 : consider as good association, as there is enough reco to cover the sim
            */
            float simMinusRecoFraction = std::max(0.f, simFraction - recoFraction);
            float score = invDenominator * simMinusRecoFraction * simMinusRecoFraction * squaredRecHitEnergy;
            simTracksterToTracksterMap->insert(simTracksterRef, recoTracksterRef, sharedEnergy, score);
          }
        }
      }

      auto sortingFunc = [](const auto& a, const auto& b) {
        if (a.score() != b.score())
          return a.score() < b.score();
        else
          return a.index() < b.index();
      };

      // Sort the maps by score in ascending order
      tracksterToSimTracksterMap->sort(sortingFunc);
      simTracksterToTracksterMap->sort(sortingFunc);

      // After populating the maps, store them in the event
      iEvent.put(std::move(tracksterToSimTracksterMap), tracksterToken.first + "To" + simTracksterToken.first);
      iEvent.put(std::move(simTracksterToTracksterMap), simTracksterToken.first + "To" + tracksterToken.first);
    }
  }
}

void AllTracksterToSimTracksterAssociatorsByHitsProducer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("allHitToTSAccoc", "allHitToTracksterAssociations");
  desc.add<std::vector<edm::InputTag>>(
      "tracksterCollections", {edm::InputTag("ticlTrackstersCLUE3DHigh"), edm::InputTag("ticlTrackstersLinks")});
  desc.add<std::vector<edm::InputTag>>(
      "simTracksterCollections", {edm::InputTag("ticlSimTracksters"), edm::InputTag("ticlSimTracksters", "fromCPs")});
  desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "RefProdVectorHGCRecHitCollection"));
  desc.add<edm::InputTag>("hitToSimClusterMap",
                          edm::InputTag("hitToSimClusterCaloParticleAssociator", "hitToSimClusterMap"));
  desc.add<edm::InputTag>("hitToCaloParticleMap",
                          edm::InputTag("hitToSimClusterCaloParticleAssociator", "hitToCaloParticleMap"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));

  descriptions.add("AllTracksterToSimTracksterAssociatorsByHitsProducer", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(AllTracksterToSimTracksterAssociatorsByHitsProducer);
