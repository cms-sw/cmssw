#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

using namespace reco;
using namespace std;

/* Constructor */

MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl::MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl(
    edm::EDProductGetter const& productGetter,
    mtd::MTDGeomUtil& geomTools,
    reco::SimToRecoCollectionMtd simToRecoMap,
    reco::RecoToSimCollectionMtd recoToSimMap)
    : productGetter_(&productGetter), geomTools_(geomTools), simToRecoMap_(simToRecoMap), recoToSimMap_(recoToSimMap) {}

//
//---member functions
//

reco::MergedRecoToSimCollectionMtd MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl::associateRecoToSim(
    const edm::Handle<FTLMergedClusterCollection>& btlRecoClusH,
    const edm::Handle<FTLMergedClusterCollection>& etlRecoClusH,
    const edm::Handle<MtdSimMergedClusterCollection>& simMergedClusH) const {
  MergedRecoToSimCollectionMtd outputCollection;

  // -- get collections
  std::array<edm::Handle<FTLMergedClusterCollection>, 2> inputRecoMergedClusH{{btlRecoClusH, etlRecoClusH}};

  const auto& simMergedClusters = *simMergedClusH.product();

  // make preliminary map: simCluster => simMergedCluster
  std::map<MtdSimLayerClusterRef, std::vector<MtdSimMergedClusterRef>> simClusToMergedMap;
  for (const auto& simMergedClus : simMergedClusters) {
    for (const auto& simClusRef : simMergedClus.clusters()) {
      simClusToMergedMap[simClusRef].push_back(
          MtdSimMergedClusterRef(simMergedClusH, &simMergedClus - &(*simMergedClusH->begin())));
    }
  }

  // loop over reco merged clusters
  for (auto const& recoMergedClusH : inputRecoMergedClusH) {
    for (const auto& detSet : *recoMergedClusH) {
      for (const auto& recoMergedClus : detSet) {
        FTLMergedClusterRef recoMergedClusterRef = edmNew::makeRefTo(recoMergedClusH, &recoMergedClus);
        std::vector<MtdSimMergedClusterRef> simClusterRefs;

        LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
            << "RecoCluster: " << recoMergedClusterRef.key()
            << " with nComponents=" << recoMergedClus.clusterRefs().size();

        // iterate over component clusters
        for (const auto& recoClusRef : recoMergedClus.clusterRefs()) {
          LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
              << "    Component id=" << recoClusRef.id() << "    key=" << recoClusRef.key();
          auto recoToSimIt = recoToSimMap_.equal_range(recoClusRef);
          if (recoToSimIt.first == recoToSimIt.second) {
            LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                << "    -> NOT FOUND in recoToSimMap)";
            if (!recoToSimMap_.empty()) {
              LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                  << "    -> First entry id = " << recoToSimMap_.begin()->first.id()
                  << " key = " << recoToSimMap_.begin()->first.key();
            }
            LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                << "  No sim clusters associated to this reco cluster";
            continue;
          }
          const auto& simClusterCandidates = (*recoToSimIt.first).second;
          for (const auto& simClusterRef : simClusterCandidates) {
            // retrieve simMergedClusters associated to this simLayerCluster
            auto const& simMergedClusters = simClusToMergedMap.find(simClusterRef);
            if (simMergedClusters == simClusToMergedMap.end()) {
              LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                  << "  No sim merged clusters associated to this sim layer cluster";
              continue;
            }
            for (const auto& simMergedClusterRef : simMergedClusters->second) {
              simClusterRefs.push_back(simMergedClusterRef);
            }
          }
        }

        // Fill output collection after removing simClusterRefs duplicates
        std::sort(simClusterRefs.begin(), simClusterRefs.end());
        simClusterRefs.erase(std::unique(simClusterRefs.begin(), simClusterRefs.end()), simClusterRefs.end());
        outputCollection.emplace_back(recoMergedClusterRef, simClusterRefs);
      }
    }
  }  // end loop over reco merged clusters

  outputCollection.post_insert();
  return outputCollection;
}

reco::MergedSimToRecoCollectionMtd MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl::associateSimToReco(
    const edm::Handle<FTLMergedClusterCollection>& btlRecoClusH,
    const edm::Handle<FTLMergedClusterCollection>& etlRecoClusH,
    const edm::Handle<MtdSimMergedClusterCollection>& simMergedClusH) const {
  MergedSimToRecoCollectionMtd outputCollection;

  // -- get the collections
  const auto& simMergedClusters = *simMergedClusH.product();
  std::array<edm::Handle<FTLMergedClusterCollection>, 2> inputH{{btlRecoClusH, etlRecoClusH}};

  // make preliminary map: recoCluster => recoMergedCluster
  std::map<FTLClusterRef, std::vector<FTLMergedClusterRef>> recoClusToMergedMap;
  for (const auto& recoMergedClusH : inputH) {
    for (const auto& detSet : *recoMergedClusH) {
      for (const auto& recoMergedClus : detSet) {
        for (const auto& recoClusRef : recoMergedClus.clusterRefs()) {
          recoClusToMergedMap[recoClusRef].push_back(edmNew::makeRefTo(recoMergedClusH, &recoMergedClus));
        }
      }
    }
  }

  // -- loop over MtdSimMergedClusters
  for (auto simMergedClusIt = simMergedClusters.begin(); simMergedClusIt != simMergedClusters.end();
       simMergedClusIt++) {
    const auto& simMergedClus = *simMergedClusIt;

    // query the simToReco map and retrieve list of reco clusters
    auto const& simMergedClusterRef =
        edm::Ref<MtdSimMergedClusterCollection>(simMergedClusH, &simMergedClus - &(*simMergedClusters.begin()));
    std::vector<FTLMergedClusterRef> recoMergedClusterRefs;

    // iterate over component clusters
    for (const auto& simClusterRef : simMergedClus.clusters()) {
      auto simToRecoIt = simToRecoMap_.equal_range(simClusterRef);
      if (simToRecoIt.first != simToRecoIt.second) {
        const auto& recoRefs = (*simToRecoIt.first).second;
        for (const auto& recoRef : recoRefs) {
          // retrieve recoMergedClusters associated to this recoCluster
          auto const& recoMergedClusters = recoClusToMergedMap.find(recoRef);

          if (recoMergedClusters == recoClusToMergedMap.end()) {
            LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                << "  No reco merged clusters associated to this reco cluster";
            continue;
          }

          for (const auto& recoMergedClusterRef : recoMergedClusters->second) {
            LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                << "  Found associated reco merged cluster: " << recoMergedClusterRef.key();
            recoMergedClusterRefs.push_back(recoMergedClusterRef);
          }
        }
      }
    }

    // Remove duplicates from recoMergedClusterRefs
    std::sort(recoMergedClusterRefs.begin(), recoMergedClusterRefs.end());
    recoMergedClusterRefs.erase(std::unique(recoMergedClusterRefs.begin(), recoMergedClusterRefs.end()),
                                recoMergedClusterRefs.end());

    outputCollection.emplace_back(simMergedClusterRef, recoMergedClusterRefs);

  }  // -- end loop over sim merged clusters

  outputCollection.post_insert();
  return outputCollection;
}
