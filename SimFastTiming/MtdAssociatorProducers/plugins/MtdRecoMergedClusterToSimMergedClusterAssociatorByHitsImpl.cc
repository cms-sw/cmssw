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
    const edm::Handle<edmNew::DetSetVector<std::vector<FTLClusterRef>>>& btlConstituentsH,
    const edm::Handle<edmNew::DetSetVector<std::vector<FTLClusterRef>>>& etlConstituentsH,
    const edm::Handle<MtdSimMergedClusterCollection>& simMergedClusH) const {
  MergedRecoToSimCollectionMtd outputCollection;

  // -- get collections
  std::array<
      std::pair<edm::Handle<FTLMergedClusterCollection>, edm::Handle<edmNew::DetSetVector<std::vector<FTLClusterRef>>>>,
      2>
      inputRecoMergedClus{{{btlRecoClusH, btlConstituentsH}, {etlRecoClusH, etlConstituentsH}}};
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
  for (auto const& [recoMergedClusH, constituentsH] : inputRecoMergedClus) {
    if (!recoMergedClusH.isValid() || !constituentsH.isValid())
      continue;
    for (const auto& detSet : *recoMergedClusH) {
      auto constIt = constituentsH->find(detSet.id());
      if (constIt == constituentsH->end()) {
        edm::LogWarning("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
            << "Missing parallel constituent DetSet for DetId: " << detSet.id();
        continue;
      }
      size_t clusterIndex = 0;
      for (const auto& recoMergedClus : detSet) {
        FTLMergedClusterRef recoMergedClusterRef = edmNew::makeRefTo(recoMergedClusH, &recoMergedClus);
        std::vector<MtdSimMergedClusterRef> simClusterRefs;
        const auto& constituentRefs = (*constIt)[clusterIndex];

        // iterate over component clusters
        for (const auto& recoClusRef : constituentRefs) {
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
        clusterIndex++;
      }
    }
  }  // end loop over reco merged clusters

  outputCollection.post_insert();
  return outputCollection;
}

reco::MergedSimToRecoCollectionMtd MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl::associateSimToReco(
    const edm::Handle<FTLMergedClusterCollection>& btlRecoClusH,
    const edm::Handle<FTLMergedClusterCollection>& etlRecoClusH,
    const edm::Handle<edmNew::DetSetVector<std::vector<FTLClusterRef>>>& btlConstituentsH,
    const edm::Handle<edmNew::DetSetVector<std::vector<FTLClusterRef>>>& etlConstituentsH,
    const edm::Handle<MtdSimMergedClusterCollection>& simMergedClusH) const {
  MergedSimToRecoCollectionMtd outputCollection;

  // -- get the collections
  const auto& simMergedClusters = *simMergedClusH.product();
  std::array<
      std::pair<edm::Handle<FTLMergedClusterCollection>, edm::Handle<edmNew::DetSetVector<std::vector<FTLClusterRef>>>>,
      2>
      inputH{{{btlRecoClusH, btlConstituentsH}, {etlRecoClusH, etlConstituentsH}}};

  // make preliminary map: recoCluster => recoMergedCluster
  std::map<FTLClusterRef, std::vector<FTLMergedClusterRef>> recoClusToMergedMap;
  for (const auto& [recoMergedClusH, constituentsH] : inputH) {
    if (!recoMergedClusH.isValid() || !constituentsH.isValid())
      continue;
    for (const auto& detSet : *recoMergedClusH) {
      auto constIt = constituentsH->find(detSet.id());
      if (constIt == constituentsH->end())
        continue;
      size_t clusterIndex = 0;
      for (const auto& recoMergedClus : detSet) {
        FTLMergedClusterRef recoMergedClusterRef = edmNew::makeRefTo(recoMergedClusH, &recoMergedClus);
        const auto& constituentRefs = (*constIt)[clusterIndex];
        for (const auto& recoClusRef : constituentRefs) {
          recoClusToMergedMap[recoClusRef].push_back(recoMergedClusterRef);
        }
        clusterIndex++;
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
