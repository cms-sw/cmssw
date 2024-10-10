// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024
#include "TracksterToSimTracksterAssociatorByHitsProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

TracksterToSimTracksterAssociatorByHitsProducer::TracksterToSimTracksterAssociatorByHitsProducer(
    const edm::ParameterSet& pset)
    : recoTracksterCollectionToken_(
          consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("tracksters"))),
      simTracksterCollectionToken_(
          consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("simTracksters"))),
      simTracksterFromCPCollectionToken_(
          consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("simTrackstersFromCP"))),
      hitToTracksterMapToken_(
          consumes<ticl::AssociationMap<ticl::mapWithFraction>>(pset.getParameter<edm::InputTag>("hitToTracksterMap"))),
      hitToSimTracksterMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToSimTracksterMap"))),
      hitToSimTracksterFromCPMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToSimTracksterFromCPMap"))),
      hitToSimClusterMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToSimClusterMap"))),
      hitToCaloParticleMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToCaloParticleMap"))),
      tracksterToHitMapToken_(
          consumes<ticl::AssociationMap<ticl::mapWithFraction>>(pset.getParameter<edm::InputTag>("tracksterToHitMap"))),
      simTracksterToHitMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("simTracksterToHitMap"))),
      simTracksterFromCPToHitMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("simTracksterFromCPToHitMap"))),
      caloParticleToken_(consumes<std::vector<CaloParticle>>(pset.getParameter<edm::InputTag>("caloParticles"))) {
  auto hitsTags = pset.getParameter<std::vector<edm::InputTag>>("hits");
  for (const auto& tag : hitsTags) {
    hitsTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  }
  produces<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      "tracksterToSimTracksterMap");
  produces<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      "simTracksterToTracksterMap");
  produces<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      "tracksterToSimTracksterFromCPMap");
  produces<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      "simTracksterFromCPToTracksterMap");
}

TracksterToSimTracksterAssociatorByHitsProducer::~TracksterToSimTracksterAssociatorByHitsProducer() {}

void TracksterToSimTracksterAssociatorByHitsProducer::produce(edm::StreamID,
                                                              edm::Event& iEvent,
                                                              const edm::EventSetup& iSetup) const {
  using namespace edm;

  Handle<std::vector<ticl::Trackster>> recoTrackstersHandle;
  iEvent.getByToken(recoTracksterCollectionToken_, recoTrackstersHandle);
  const auto& recoTracksters = *recoTrackstersHandle;

  Handle<std::vector<ticl::Trackster>> simTrackstersHandle;
  iEvent.getByToken(simTracksterCollectionToken_, simTrackstersHandle);
  const auto& simTracksters = *simTrackstersHandle;

  Handle<std::vector<ticl::Trackster>> simTrackstersFromCPHandle;
  iEvent.getByToken(simTracksterFromCPCollectionToken_, simTrackstersFromCPHandle);
  const auto& simTrackstersFromCP = *simTrackstersFromCPHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToTracksterMapHandle;
  iEvent.getByToken(hitToTracksterMapToken_, hitToTracksterMapHandle);
  const auto& hitToTracksterMap = *hitToTracksterMapHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimTracksterMapHandle;
  iEvent.getByToken(hitToSimTracksterMapToken_, hitToSimTracksterMapHandle);
  const auto& hitToSimTracksterMap = *hitToSimTracksterMapHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimTracksterFromCPMapHandle;
  iEvent.getByToken(hitToSimTracksterFromCPMapToken_, hitToSimTracksterFromCPMapHandle);
  const auto& hitToSimTracksterFromCPMap = *hitToSimTracksterFromCPMapHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimClusterMapHandle;
  iEvent.getByToken(hitToSimClusterMapToken_, hitToSimClusterMapHandle);
  const auto& hitToSimClusterMap = *hitToSimClusterMapHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> hitToCaloParticleMapHandle;
  iEvent.getByToken(hitToCaloParticleMapToken_, hitToCaloParticleMapHandle);
  const auto& hitToCaloParticleMap = *hitToCaloParticleMapHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> tracksterToHitMapHandle;
  iEvent.getByToken(tracksterToHitMapToken_, tracksterToHitMapHandle);
  const auto& tracksterToHitMap = *tracksterToHitMapHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> simTracksterToHitMapHandle;
  iEvent.getByToken(simTracksterToHitMapToken_, simTracksterToHitMapHandle);
  const auto& simTracksterToHitMap = *simTracksterToHitMapHandle;

  Handle<ticl::AssociationMap<ticl::mapWithFraction>> simTracksterFromCPToHitMapHandle;
  iEvent.getByToken(simTracksterFromCPToHitMapToken_, simTracksterFromCPToHitMapHandle);
  const auto& simTracksterFromCPToHitMap = *simTracksterFromCPToHitMapHandle;

  Handle<std::vector<CaloParticle>> caloParticlesHandle;
  iEvent.getByToken(caloParticleToken_, caloParticlesHandle);

  MultiVectorManager<HGCRecHit> rechitManager;
  for (const auto& token : hitsTokens_) {
    Handle<HGCRecHitCollection> hitsHandle;
    iEvent.getByToken(token, hitsHandle);
    rechitManager.addVector(*hitsHandle);
  }

  auto tracksterToSimTracksterMap = std::make_unique<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      recoTrackstersHandle, simTrackstersHandle, iEvent);
  auto tracksterToSimTracksterFromCPMap = std::make_unique<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      recoTrackstersHandle, simTrackstersFromCPHandle, iEvent);

  auto simTracksterToTracksterMap = std::make_unique<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      simTrackstersHandle, recoTrackstersHandle, iEvent);
  auto simTracksterFromCPToTracksterMap = std::make_unique<
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>>(
      simTrackstersFromCPHandle, recoTrackstersHandle, iEvent);
  for (unsigned int tracksterIndex = 0; tracksterIndex < recoTracksters.size(); ++tracksterIndex) {
    edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, tracksterIndex);
    float recoToSimScoresDenominator = 0.f;
    const auto& recoTracksterHitsAndFractions = tracksterToHitMap[tracksterIndex];
    ticl::AssociationMap<ticl::mapWithFraction> hitToAssociatedSimTracksterMap(recoTracksterHitsAndFractions.size());
    std::vector<unsigned int> associatedSimTracksterIndices;
    ticl::AssociationMap<ticl::mapWithFraction> hitToAssociatedSimTracksterFromCPMap(
        recoTracksterHitsAndFractions.size());
    std::vector<unsigned int> associatedSimTracksterFromCPIndices;
    for (unsigned int i = 0; i < recoTracksterHitsAndFractions.size(); ++i) {
      const auto& [hitIndex, recoFraction] = recoTracksterHitsAndFractions[i];
      const auto& recHit = rechitManager[hitIndex];
      float squaredRecoFraction = recoFraction * recoFraction;
      float rechitEnergy = recHit.energy();
      float squaredRecHitEnergy = rechitEnergy * rechitEnergy;
      recoToSimScoresDenominator += squaredRecoFraction * squaredRecHitEnergy;

      const auto& hitToSimTracksterVec = hitToSimTracksterMap[hitIndex];
      for (const auto& [simTracksterIndex, fraction] : hitToSimTracksterVec) {
        const auto& simTrackster = simTracksters[simTracksterIndex];
        auto& seed = simTrackster.seedID();
        float simFraction = 0;
        if (seed == caloParticlesHandle.id()) {
          unsigned int caloParticleIndex = simTrackster.seedIndex();
          auto it = std::find_if(hitToCaloParticleMap[hitIndex].begin(),
                                 hitToCaloParticleMap[hitIndex].end(),
                                 [caloParticleIndex](const auto& pair) { return pair.first == caloParticleIndex; });
          if (it != hitToCaloParticleMap[hitIndex].end()) {
            simFraction = it->second;
          }
        } else {
          unsigned int simClusterIndex = simTracksters[simTracksterIndex].seedIndex();
          auto it = std::find_if(hitToSimClusterMap[hitIndex].begin(),
                                 hitToSimClusterMap[hitIndex].end(),
                                 [simClusterIndex](const auto& pair) { return pair.first == simClusterIndex; });
          if (it != hitToSimClusterMap[hitIndex].end()) {
            simFraction = it->second;
          }
        }
        hitToAssociatedSimTracksterMap.insert(i, simTracksterIndex, simFraction);
        associatedSimTracksterIndices.push_back(simTracksterIndex);
      }

      // do the same for caloparticles and simTracksterFromCP
      const auto& hitToSimTracksterFromCPVec = hitToSimTracksterFromCPMap[hitIndex];
      for (const auto& [simTracksterIndex, simFraction] : hitToSimTracksterFromCPVec) {
        unsigned int caloParticleIndex = simTracksters[simTracksterIndex].seedIndex();
        float caloParticleFraction = 0;
        auto it = std::find_if(hitToCaloParticleMap[hitIndex].begin(),
                               hitToCaloParticleMap[hitIndex].end(),
                               [caloParticleIndex](const auto& pair) { return pair.first == caloParticleIndex; });
        if (it != hitToCaloParticleMap[hitIndex].end()) {
          caloParticleFraction = it->second;
        }
        hitToAssociatedSimTracksterFromCPMap.insert(i, simTracksterIndex, caloParticleFraction);
        associatedSimTracksterFromCPIndices.push_back(simTracksterIndex);
      }
    }
    std::sort(associatedSimTracksterIndices.begin(), associatedSimTracksterIndices.end());
    associatedSimTracksterIndices.erase(
        std::unique(associatedSimTracksterIndices.begin(), associatedSimTracksterIndices.end()),
        associatedSimTracksterIndices.end());

    std::sort(associatedSimTracksterFromCPIndices.begin(), associatedSimTracksterFromCPIndices.end());
    associatedSimTracksterFromCPIndices.erase(
        std::unique(associatedSimTracksterFromCPIndices.begin(), associatedSimTracksterFromCPIndices.end()),
        associatedSimTracksterFromCPIndices.end());

    // Add missing sim tracksters with 0 shared energy to hitToAssociatedSimTracksterMap and hitToAssociatedSimTracksterFromCPMap
    for (unsigned int i = 0; i < recoTracksterHitsAndFractions.size(); ++i) {
      unsigned int hitId = recoTracksterHitsAndFractions[i].first;
      const auto& simTracksterVec = hitToSimTracksterMap[hitId];
      for (unsigned int simTracksterIndex : associatedSimTracksterIndices) {
        if (std::find_if(simTracksterVec.begin(), simTracksterVec.end(), [simTracksterIndex](const auto& pair) {
              return pair.first == simTracksterIndex;
            }) == simTracksterVec.end()) {
          hitToAssociatedSimTracksterMap.insert(i, simTracksterIndex, 0);
        }
      }

      const auto& simTracksterFromCPVec = hitToSimTracksterFromCPMap[hitId];
      for (unsigned int simTracksterIndex : associatedSimTracksterFromCPIndices) {
        if (std::find_if(
                simTracksterFromCPVec.begin(), simTracksterFromCPVec.end(), [simTracksterIndex](const auto& pair) {
                  return pair.first == simTracksterIndex;
                }) == simTracksterFromCPVec.end()) {
          hitToAssociatedSimTracksterFromCPMap.insert(i, simTracksterIndex, 0);
        }
      }
    }

    const float invDenominator = 1.f / recoToSimScoresDenominator;

    for (unsigned int i = 0; i < recoTracksterHitsAndFractions.size(); ++i) {
      unsigned int hitIndex = recoTracksterHitsAndFractions[i].first;
      const auto& recHit = rechitManager[hitIndex];
      float recoFraction = recoTracksterHitsAndFractions[i].second;
      float squaredRecoFraction = recoFraction * recoFraction;
      float squaredRecHitEnergy = recHit.energy() * recHit.energy();
      float recoSharedEnergy = recHit.energy() * recoFraction;
      const auto& simTracksterVec = hitToAssociatedSimTracksterMap[i];
      for (const auto& [simTracksterIndex, simFraction] : simTracksterVec) {
        edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersHandle, simTracksterIndex);
        float sharedEnergy = std::min(simFraction * recHit.energy(), recoSharedEnergy);
        float squaredFraction =
            std::min(squaredRecoFraction, (recoFraction - simFraction) * (recoFraction - simFraction));
        float score = invDenominator * squaredFraction * squaredRecHitEnergy;
        tracksterToSimTracksterMap->insert(recoTracksterRef, simTracksterRef, sharedEnergy, score);
      }

      const auto& simTracksterFromCPVec = hitToAssociatedSimTracksterFromCPMap[i];
      for (const auto& [simTracksterIndex, simFraction] : simTracksterFromCPVec) {
        edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersFromCPHandle, simTracksterIndex);
        float sharedEnergy = std::min(simFraction * recHit.energy(), recoSharedEnergy);
        float squaredFraction =
            std::min(squaredRecoFraction, (recoFraction - simFraction) * (recoFraction - simFraction));
        float score = invDenominator * squaredFraction * squaredRecHitEnergy;
        tracksterToSimTracksterFromCPMap->insert(recoTracksterRef, simTracksterRef, sharedEnergy, score);
      }
    }
  }

  // Reverse mapping: SimTrackster -> RecoTrackster
  for (unsigned int tracksterIndex = 0; tracksterIndex < simTracksters.size(); ++tracksterIndex) {
    edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersHandle, tracksterIndex);
    float simToRecoScoresDenominator = 0.f;
    const auto& simTracksterHitsAndFractions = simTracksterToHitMap[tracksterIndex];
    ticl::AssociationMap<ticl::mapWithFraction> hitToAssociatedRecoTracksterMap(simTracksterHitsAndFractions.size());
    std::vector<unsigned int> associatedRecoTracksterIndices;
    const auto& simTrackster = simTracksters[tracksterIndex];
    auto& seed = simTrackster.seedID();
    unsigned int simObjectIndex = simTrackster.seedIndex();
    bool isSimTracksterFromCP = (seed == caloParticlesHandle.id());
    std::vector<float> simFractions(simTracksterHitsAndFractions.size(), 0.f);
    for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
      const auto& [hitIndex, simTracksterFraction] = simTracksterHitsAndFractions[i];

      auto it = isSimTracksterFromCP
                    ? (std::find_if(hitToCaloParticleMap[hitIndex].begin(),
                                    hitToCaloParticleMap[hitIndex].end(),
                                    [simObjectIndex](const auto& pair) { return pair.first == simObjectIndex; }))
                    : std::find_if(hitToSimClusterMap[hitIndex].begin(),
                                   hitToSimClusterMap[hitIndex].end(),
                                   [simObjectIndex](const auto& pair) { return pair.first == simObjectIndex; });
      if (it != hitToCaloParticleMap[hitIndex].end() and it != hitToSimClusterMap[hitIndex].end()) {
        simFractions[i] = it->second;
      }
      float simFraction = simFractions[i];
      const auto& recHit = rechitManager[hitIndex];
      float squaredSimFraction = simFraction * simFraction;
      float squaredRecHitEnergy = recHit.energy() * recHit.energy();
      simToRecoScoresDenominator += squaredSimFraction * squaredRecHitEnergy;

      const auto& hitToRecoTracksterVec = hitToTracksterMap[hitIndex];
      for (const auto& [recoTracksterIndex, recoFraction] : hitToRecoTracksterVec) {
        hitToAssociatedRecoTracksterMap.insert(i, recoTracksterIndex, recoFraction);
        associatedRecoTracksterIndices.push_back(recoTracksterIndex);
      }
    }

    std::sort(associatedRecoTracksterIndices.begin(), associatedRecoTracksterIndices.end());
    associatedRecoTracksterIndices.erase(
        std::unique(associatedRecoTracksterIndices.begin(), associatedRecoTracksterIndices.end()),
        associatedRecoTracksterIndices.end());

    for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
      unsigned int hitIndex = simTracksterHitsAndFractions[i].first;
      const auto& hitToRecoTracksterVec = hitToTracksterMap[hitIndex];
      for (unsigned int recoTracksterIndex : associatedRecoTracksterIndices) {
        if (std::find_if(
                hitToRecoTracksterVec.begin(), hitToRecoTracksterVec.end(), [recoTracksterIndex](const auto& pair) {
                  return pair.first == recoTracksterIndex;
                }) == hitToRecoTracksterVec.end()) {
          hitToAssociatedRecoTracksterMap.insert(i, recoTracksterIndex, 0);
        }
      }
    }

    const float invDenominator = 1.f / simToRecoScoresDenominator;

    for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
      const auto& [hitIndex, simTracksterFraction] = simTracksterHitsAndFractions[i];
      float simFraction = simFractions[i];
      const auto& recHit = rechitManager[hitIndex];
      float squaredSimFraction = simFraction * simFraction;
      float squaredRecHitEnergy = recHit.energy() * recHit.energy();
      float simSharedEnergy = recHit.energy() * simFraction;

      const auto& hitToRecoTracksterVec = hitToAssociatedRecoTracksterMap[i];
      for (const auto& [recoTracksterIndex, recoFraction] : hitToRecoTracksterVec) {
        edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, recoTracksterIndex);
        float sharedEnergy = std::min(recoFraction * recHit.energy(), simSharedEnergy);
        float squaredFraction =
            std::min(squaredSimFraction, (recoFraction - simFraction) * (recoFraction - simFraction));
        float score = invDenominator * squaredFraction * squaredRecHitEnergy;
        simTracksterToTracksterMap->insert(simTracksterRef, recoTracksterRef, sharedEnergy, score);
      }
    }
  }

  // Repeat the reverse mapping process for SimTracksterFromCP
  for (unsigned int tracksterIndex = 0; tracksterIndex < simTrackstersFromCP.size(); ++tracksterIndex) {
    edm::Ref<std::vector<ticl::Trackster>> simTracksterRef(simTrackstersFromCPHandle, tracksterIndex);
    float simToRecoScoresDenominator = 0.f;
    const auto& simTracksterHitsAndFractions = simTracksterFromCPToHitMap[tracksterIndex];
    ticl::AssociationMap<ticl::mapWithFraction> hitToAssociatedRecoTracksterMap(simTracksterHitsAndFractions.size());
    std::vector<unsigned int> associatedRecoTracksterIndices;
    std::vector<float> simFractions(simTracksterHitsAndFractions.size(), 0.f);
    const auto& simTrackster = simTrackstersFromCP[tracksterIndex];
    unsigned int simObjectIndex = simTrackster.seedIndex();
    for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
      const auto& [hitIndex, simTracksterFraction] = simTracksterHitsAndFractions[i];
      auto it = std::find_if(hitToCaloParticleMap[hitIndex].begin(),
                             hitToCaloParticleMap[hitIndex].end(),
                             [simObjectIndex](const auto& pair) { return pair.first == simObjectIndex; });
      if (it != hitToCaloParticleMap[hitIndex].end()) {
        simFractions[i] = it->second;
      }
      float simFraction = simFractions[i];

      const auto& recHit = rechitManager[hitIndex];
      float squaredSimFraction = simFraction * simFraction;
      float squaredRecHitEnergy = recHit.energy() * recHit.energy();
      simToRecoScoresDenominator += squaredSimFraction * squaredRecHitEnergy;

      const auto& hitToRecoTracksterVec = hitToTracksterMap[hitIndex];
      for (const auto& [recoTracksterIndex, recoFraction] : hitToRecoTracksterVec) {
        hitToAssociatedRecoTracksterMap.insert(i, recoTracksterIndex, recoFraction);
        associatedRecoTracksterIndices.push_back(recoTracksterIndex);
      }
    }

    std::sort(associatedRecoTracksterIndices.begin(), associatedRecoTracksterIndices.end());
    associatedRecoTracksterIndices.erase(
        std::unique(associatedRecoTracksterIndices.begin(), associatedRecoTracksterIndices.end()),
        associatedRecoTracksterIndices.end());

    for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
      unsigned int hitIndex = simTracksterHitsAndFractions[i].first;
      const auto& hitToRecoTracksterVec = hitToTracksterMap[hitIndex];
      for (unsigned int recoTracksterIndex : associatedRecoTracksterIndices) {
        if (std::find_if(
                hitToRecoTracksterVec.begin(), hitToRecoTracksterVec.end(), [recoTracksterIndex](const auto& pair) {
                  return pair.first == recoTracksterIndex;
                }) == hitToRecoTracksterVec.end()) {
          hitToAssociatedRecoTracksterMap.insert(i, recoTracksterIndex, 0.f);
        }
      }
    }

    const float invDenominator = 1.f / simToRecoScoresDenominator;

    for (unsigned int i = 0; i < simTracksterHitsAndFractions.size(); ++i) {
      const auto& [hitIndex, simTracksterFraction] = simTracksterHitsAndFractions[i];
      const auto& recHit = rechitManager[hitIndex];
      float simFraction = simFractions[i];
      float squaredSimFraction = simFraction * simFraction;
      float squaredRecHitEnergy = recHit.energy() * recHit.energy();
      float simSharedEnergy = recHit.energy() * simFraction;

      const auto& hitToRecoTracksterVec = hitToAssociatedRecoTracksterMap[i];
      for (const auto& [recoTracksterIndex, recoFraction] : hitToRecoTracksterVec) {
        edm::Ref<std::vector<ticl::Trackster>> recoTracksterRef(recoTrackstersHandle, recoTracksterIndex);
        float sharedEnergy = std::min(recoFraction * recHit.energy(), simSharedEnergy);
        float squaredFraction =
            std::min(squaredSimFraction, (recoFraction - simFraction) * (recoFraction - simFraction));
        float score = invDenominator * squaredFraction * squaredRecHitEnergy;
        simTracksterFromCPToTracksterMap->insert(simTracksterRef, recoTracksterRef, sharedEnergy, score);
      }
    }
  }
  tracksterToSimTracksterMap->sort(true);
  tracksterToSimTracksterFromCPMap->sort(true);
  simTracksterToTracksterMap->sort(true);
  simTracksterFromCPToTracksterMap->sort(true);

  iEvent.put(std::move(tracksterToSimTracksterMap), "tracksterToSimTracksterMap");
  iEvent.put(std::move(tracksterToSimTracksterFromCPMap), "tracksterToSimTracksterFromCPMap");
  iEvent.put(std::move(simTracksterToTracksterMap), "simTracksterToTracksterMap");
  iEvent.put(std::move(simTracksterFromCPToTracksterMap), "simTracksterFromCPToTracksterMap");
}

void TracksterToSimTracksterAssociatorByHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersMerge"));
  desc.add<edm::InputTag>("simTracksters", edm::InputTag("ticlSimTracksters"));
  desc.add<edm::InputTag>("simTrackstersFromCP", edm::InputTag("ticlSimTracksters", "fromCPs"));

  desc.add<edm::InputTag>("hitToTracksterMap", edm::InputTag("hitToTracksterAssociator", "hitToTracksterMap"));
  desc.add<edm::InputTag>("hitToSimTracksterMap",
                          edm::InputTag("allHitToTracksterAssociations", "hitToticlSimTracksters"));
  desc.add<edm::InputTag>("hitToSimTracksterFromCPMap",
                          edm::InputTag("allHitToTracksterAssociations", "hitToticlSimTrackstersfromCPs"));
  desc.add<edm::InputTag>("hitToSimClusterMap",
                          edm::InputTag("hitToSimClusterCaloParticleAssociator", "hitToSimClusterMap"));
  desc.add<edm::InputTag>("hitToCaloParticleMap",
                          edm::InputTag("hitToSimClusterCaloParticleAssociator", "hitToCaloParticleMap"));
  desc.add<edm::InputTag>("tracksterToHitMap", edm::InputTag("hitToTrackstersAssociationPR", "tracksterToHitMap"));
  desc.add<edm::InputTag>("simTracksterToHitMap",
                          edm::InputTag("allHitToTracksterAssociations", "ticlSimTrackstersToHit"));
  desc.add<edm::InputTag>("simTracksterFromCPToHitMap",
                          edm::InputTag("allHitToTracksterAssociations", "ticlSimTrackstersfromCPsToHit"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));

  desc.add<std::vector<edm::InputTag>>("hits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  descriptions.add("tracksterToSimTracksterAssociatorByHitsProducer", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(TracksterToSimTracksterAssociatorByHitsProducer);