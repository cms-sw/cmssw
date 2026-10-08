//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "MtdSimMergedClusterToTPAssociatorByTrackIdImpl.h"

using namespace reco;
using namespace std;

/* Constructor */
MtdSimMergedClusterToTPAssociatorByTrackIdImpl::MtdSimMergedClusterToTPAssociatorByTrackIdImpl(
    edm::EDProductGetter const& productGetter,
    reco::SimToTPCollectionMtd simToTPMap,
    reco::TPToSimCollectionMtd tpToSimMap)
    : productGetter_(&productGetter), simToTPMap_(simToTPMap), tpToSimMap_(tpToSimMap) {}

//
//---member functions
//

reco::MergedSimToTPCollectionMtd MtdSimMergedClusterToTPAssociatorByTrackIdImpl::associateSimToTP(
    const edm::Handle<MtdSimMergedClusterCollection>& simMergedClusH,
    const edm::Handle<TrackingParticleCollection>& trackingParticleH) const {
  MergedSimToTPCollectionMtd outputCollection(productGetter_);

  const auto& simMergedClusters = *simMergedClusH.product();
  std::vector<TrackingParticleRef> tpRefs;

  for (auto simMergedClusIt = simMergedClusters.begin(); simMergedClusIt != simMergedClusters.end();
       simMergedClusIt++) {
    const auto& simMergedClus = *simMergedClusIt;
    MtdSimMergedClusterRef simMergedClusterRef =
        edm::Ref<MtdSimMergedClusterCollection>(simMergedClusH, &simMergedClus - &(*simMergedClusH->begin()));

    std::vector<TrackingParticleRef> associatedTPs;

    LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
        << "Iterating on SimMergedCluster #" << simMergedClusterRef.key() << " with E=" << simMergedClus.simEnergy()
        << " MeV"
        << ", containing #" << simMergedClus.clusters().size() << "sim clusters" << std::endl;
    // iterate over component sim clusters
    for (const auto& simClusterRef : simMergedClus.clusters()) {
      // query the simCluster -> TP map to find associated tracking particles
      LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
          << "  Iterating on component simCluster #" << simClusterRef.key() << " with E=" << simClusterRef->simEnergy()
          << " MeV" << std::endl;
      auto simToTPIt = simToTPMap_.find(simClusterRef);
      if (simToTPIt != simToTPMap_.end()) {
        LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
            << "    Found " << std::distance(simToTPIt->val.begin(), simToTPIt->val.end())
            << " TP matches to simCluster with E=" << simClusterRef->simEnergy() << " MeV" << std::endl;
        for (const auto& tpRef : simToTPIt->val) {
          LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
              << "    Found associated TP with pdgId=" << tpRef->pdgId() << ", pt=" << tpRef->pt()
              << " GeV, eta=" << tpRef->eta() << ", phi=" << tpRef->phi() << std::endl;
          associatedTPs.push_back(tpRef);
        }
      } else {
        LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
            << "    No TP matches found for simCluster with E=" << simClusterRef->simEnergy() << " MeV" << std::endl;
      }
    }

    // remove duplicates from associatedTPs and save to map
    std::sort(associatedTPs.begin(), associatedTPs.end());
    associatedTPs.erase(std::unique(associatedTPs.begin(), associatedTPs.end()), associatedTPs.end());
    for (auto const& tpRef : associatedTPs)
      outputCollection.insert(simMergedClusterRef, tpRef);
  }

  return outputCollection;
}

reco::TPToMergedSimCollectionMtd MtdSimMergedClusterToTPAssociatorByTrackIdImpl::associateTPToSim(
    const edm::Handle<MtdSimMergedClusterCollection>& simMergedClusH,
    const edm::Handle<TrackingParticleCollection>& trackingParticleH) const {
  TPToMergedSimCollectionMtd outputCollection(productGetter_);

  // -- get the collections
  const auto& simMergedClusters = *simMergedClusH.product();
  const auto& trackingParticles = *trackingParticleH.product();

  // make preliminary map: simCluster => simMergedCluster
  std::map<MtdSimLayerClusterRef, std::vector<MtdSimMergedClusterRef>> simClusToMergedMap;
  for (const auto& simMergedClus : simMergedClusters) {
    for (const auto& simClusRef : simMergedClus.clusters()) {
      simClusToMergedMap[simClusRef].push_back(
          MtdSimMergedClusterRef(simMergedClusH, &simMergedClus - &(*simMergedClusH->begin())));
    }
  }

  for (auto tpIt = trackingParticles.begin(); tpIt != trackingParticles.end(); tpIt++) {
    size_t tpIndex = tpIt - trackingParticles.begin();
    TrackingParticleRef tpRef = edm::Ref<TrackingParticleCollection>(trackingParticleH, tpIndex);
    std::vector<MtdSimMergedClusterRef> simMergedClusRefs;

    LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
        << "Looking for matches to TP #" << tpRef.key() << " with pt=" << tpIt->pt() << " GeV, eta=" << tpIt->eta()
        << ", phi=" << tpIt->phi() << std::endl;

    // find SimClus associated to TPs
    auto TPtoSimIt = tpToSimMap_.find(tpRef);
    if (TPtoSimIt != tpToSimMap_.end()) {
      LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
          << "  Found " << std::distance(TPtoSimIt->val.begin(), TPtoSimIt->val.end())
          << " simCluster matches to this TP." << std::endl;
      for (auto simClusRef : TPtoSimIt->val) {
        // find simMergedCluster it belongs to through preliminary map
        auto const& simMergedClusters = simClusToMergedMap.find(simClusRef);
        if (simMergedClusters == simClusToMergedMap.end()) {
          LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
              << "WARNING: no simMergedCluster found for simClusterRef with key " << simClusRef.key() << std::endl;
          continue;
        }
        for (const auto& simMergedClusterRef : simMergedClusters->second) {
          LogDebug("MtdSimMergedClusterToTPAssociatorByTrackIdImpl")
              << "  Found associated SimMergedCluster #" << simMergedClusterRef.key()
              << " with E=" << simMergedClusterRef->simEnergy() << " MeV" << std::endl;
          simMergedClusRefs.push_back(simMergedClusterRef);
        }
      }
    }

    // remove duplicates from simMergedClusRefs and save to map
    std::sort(simMergedClusRefs.begin(), simMergedClusRefs.end());
    simMergedClusRefs.erase(std::unique(simMergedClusRefs.begin(), simMergedClusRefs.end()), simMergedClusRefs.end());
    for (auto const& simMergedClusterRef : simMergedClusRefs)
      outputCollection.insert(tpRef, simMergedClusterRef);
  }

  // save final output
  return outputCollection;
}
