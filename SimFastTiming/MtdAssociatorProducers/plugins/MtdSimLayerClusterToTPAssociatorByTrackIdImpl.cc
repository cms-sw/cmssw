//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "MtdSimLayerClusterToTPAssociatorByTrackIdImpl.h"

using namespace reco;
using namespace std;

/* Constructor */
MtdSimLayerClusterToTPAssociatorByTrackIdImpl::MtdSimLayerClusterToTPAssociatorByTrackIdImpl(
    edm::EDProductGetter const& productGetter)
    : productGetter_(&productGetter) {}

//
//---member functions
//

reco::SimToTPCollectionMtd MtdSimLayerClusterToTPAssociatorByTrackIdImpl::associateSimToTP(
    const edm::Handle<MtdSimLayerClusterCollection>& simClusH,
    const edm::Handle<TrackingParticleCollection>& trackingParticleH) const {
  SimToTPCollectionMtd outputCollection(productGetter_);

  // -- get the collections
  const auto& simClusters = *simClusH.product();
  const auto& trackingParticles = *trackingParticleH.product();

  // -- Loop over tracking particles and build a temporary map of trackId, eventId  --> tpRef
  std::map<std::pair<unsigned int, uint32_t>, TrackingParticleRef> tpIdMap;
  for (auto tpIt = trackingParticles.begin(); tpIt != trackingParticles.end(); tpIt++) {
    const auto& tp = *tpIt;
    EncodedEventId tpEventId = tp.eventId();
    for (unsigned int igt = 0; igt < tp.g4Tracks().size(); igt++) {
      unsigned int tpTrackId = tp.g4Tracks()[igt].trackId();
      TrackingParticleRef tpRef =
          edm::Ref<TrackingParticleCollection>(trackingParticleH, tpIt - trackingParticles.begin());
      tpIdMap[std::make_pair(tpTrackId, tpEventId.rawId())] = tpRef;
    }
  }

  // -- loop over sim clusters and get the trackId, eventId

  LogDebug("MtdSimLayerClusterToTPAssociator")
      << " Found " << simClusters.size() << " MtdSimLayerClusters in the event";

  for (auto simClusIt = simClusters.begin(); simClusIt != simClusters.end(); simClusIt++) {
    const auto& simClus = *simClusIt;
    size_t simClusIndex = simClusIt - simClusters.begin();
    MtdSimLayerClusterRef simClusterRef = edm::Ref<MtdSimLayerClusterCollection>(simClusH, simClusIndex);
    EncodedEventId simClusEventId = simClus.eventId();
    for (unsigned int igt = 0; igt < simClus.g4Tracks().size(); igt++) {
      unsigned int simClusTrackId = simClus.g4Tracks()[igt].trackId();
      std::pair uniqueId = std::make_pair(simClusTrackId, simClusEventId.rawId());
      auto it = tpIdMap.find(uniqueId);

      if (it != tpIdMap.end()) {
        TrackingParticleRef tpRef = tpIdMap[uniqueId];
        outputCollection.insert(simClusterRef, tpRef);

        LogDebug("MtdSimLayerClusterToTPAssociator::associateSimToTP")
            << "MtdSimLayerCluster: index = " << simClusIndex << "   simClus TrackId = " << simClusTrackId
            << " simClus EventId = " << simClusEventId.rawId() << " simClus Eta = " << simClus.eta()
            << " simClus Phi = " << simClus.phi() << "  simClus Time = " << simClus.simLCTime()
            << "  simClus Energy = " << simClus.simLCEnergy() << std::endl;
        LogDebug("MtdSimLayerClusterToTPAssociator::associateSimToTP")
            << "  --> Found associated tracking particle:  tp TrackId = " << (*tpRef).g4Tracks()[0].trackId()
            << " tp EventId = " << (*tpRef).eventId().rawId() << std::endl;
      }
    }
  }  // -- end loop over sim clus

  return outputCollection;
}

reco::TPToSimCollectionMtd MtdSimLayerClusterToTPAssociatorByTrackIdImpl::associateTPToSim(
    const edm::Handle<MtdSimLayerClusterCollection>& simClusH,
    const edm::Handle<TrackingParticleCollection>& trackingParticleH) const {
  TPToSimCollectionMtd outputCollection(productGetter_);

  // -- get the collections
  const auto& simClusters = *simClusH.product();
  const auto& trackingParticles = *trackingParticleH.product();

  // -- Loop over MtdSimLayerClusters and build a temporary map of trackId, eventId --> simClusterRef
  std::map<std::pair<unsigned int, uint32_t>, std::vector<MtdSimLayerClusterRef>> simClusIdMap;
  for (auto simClusIt = simClusters.begin(); simClusIt != simClusters.end(); simClusIt++) {
    const auto& simClus = *simClusIt;
    EncodedEventId simClusEventId = simClus.eventId();
    for (unsigned int igt = 0; igt < simClus.g4Tracks().size(); igt++) {
      unsigned int simClusTrackId = simClus.g4Tracks()[igt].trackId();
      MtdSimLayerClusterRef simClusterRef =
          edm::Ref<MtdSimLayerClusterCollection>(simClusH, simClusIt - simClusters.begin());
      simClusIdMap[std::make_pair(simClusTrackId, simClusEventId.rawId())].push_back(simClusterRef);
    }
  }

  // -- Loop over the tracking particles
  for (auto tpIt = trackingParticles.begin(); tpIt != trackingParticles.end(); tpIt++) {
    const auto& tp = *tpIt;
    size_t tpIndex = tpIt - trackingParticles.begin();
    TrackingParticleRef tpRef = edm::Ref<TrackingParticleCollection>(trackingParticleH, tpIndex);
    EncodedEventId tpEventId = tp.eventId();
    for (unsigned int igt = 0; igt < tp.g4Tracks().size(); igt++) {
      unsigned int tpTrackId = tp.g4Tracks()[igt].trackId();
      std::pair uniqueId = std::make_pair(tpTrackId, tpEventId.rawId());
      auto it = simClusIdMap.find(uniqueId);

      if (it != simClusIdMap.end()) {
        for (unsigned int i = 0; i < simClusIdMap[uniqueId].size(); i++) {
          MtdSimLayerClusterRef simClusterRef = simClusIdMap[uniqueId][i];

          outputCollection.insert(tpRef, simClusterRef);

          LogDebug("MtdSimLayerClusterToTPAssociator")
              << "Tracking particle:  index = " << tpIndex << "  tp TrackId = " << tpTrackId
              << "  tp EventId = " << tpEventId.rawId();
          LogDebug("MtdSimLayerClusterToTPAssociator")
              << " --> Found associated MtdSimLayerCluster:  simClus TrackId = "
              << (*simClusterRef).g4Tracks()[0].trackId() << " simClus EventId = " << (*simClusterRef).eventId().rawId()
              << " simClus Eta = " << (*simClusterRef).eta() << " simClus Phi = " << (*simClusterRef).phi()
              << "  simClus Time = " << (*simClusterRef).simLCTime()
              << "  simClus Energy = " << (*simClusterRef).simLCEnergy() << std::endl;
        }
      }
    }
  }

  return outputCollection;
}
