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

  // make preliminary map: detId -> SimMergedClusterRef

  std::map<uint32_t, std::vector<MtdSimMergedClusterRef>> simMergedClusIdsMap;
  for (auto simClusIt = simMergedClusters.begin(); simClusIt != simMergedClusters.end(); simClusIt++) {
    const auto& simClus = *simClusIt;

    edm::Ref<MtdSimMergedClusterCollection> simClusterRef =
        edm::Ref<MtdSimMergedClusterCollection>(simMergedClusH, simClusIt - simMergedClusters.begin());

    std::vector<DetId> detIds = simClus.detIds();

    for (const auto& detId : detIds) {
      uint32_t modId = geomTools_.sensorModuleId(detId);
      simMergedClusIdsMap[modId].push_back(simClusterRef);
      LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
          << "Considered SimMergedCluster detId = " << modId << std::endl;
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
            << " with size=" << recoMergedClus.size();

        LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
            << "Reco cluster : " << recoMergedClus.id();

        std::vector<uint64_t> recoMergedClusHitIds(recoMergedClus.size());
        std::vector<DetId> recodetIds = recoMergedClus.clusterIds();
        // -- loop over hits in the reco cluster and find their unique ids
        for (size_t ihit = 0; ihit < recoMergedClus.size(); ++ihit) {
          // -- Get an unique id from sensor module detId , row, column
          uint64_t uniqueId = recoMergedClus.hUniqueId(ihit);
          recoMergedClusHitIds[ihit] = uniqueId;

          LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
              << "  reco mergedcluster hit uniqueId : " << uniqueId;
        }
        for (const auto& recodetId : recodetIds) {
          uint32_t recoModId = geomTools_.sensorModuleId(recodetId);
          for (const auto& simMergedClusterRef : simMergedClusIdsMap[recoModId]) {
            const auto& simMergedClus = *simMergedClusterRef;
            //std::vector<uint64_t> simMergedClusHitIds = simMergedClus.hitUniqueIds();
            std::vector<uint64_t> simMergedClusHitIds;
            for (const auto& simLayerClus : simMergedClus.clusters()) {
              for (const auto& hit : simLayerClus->detIds_and_rows()) {
                // Force the hit to map to the macroscopic module ID
                uint32_t modId = geomTools_.sensorModuleId(hit.first);
                
                // Compress the row and column
                uint8_t rowcol = static_cast<uint8_t>((std::clamp(static_cast<int>(hit.second.first), 0, 15) << 4) | 
                                                       std::clamp(static_cast<int>(hit.second.second), 0, 15));
                
                // Construct the 64-bit ID matching the Reco format
                simMergedClusHitIds.push_back((static_cast<uint64_t>(modId) << 8) | static_cast<uint64_t>(rowcol));
              }
            }
            
            // Ensure the generated list is sorted for std::set_intersection
            std::sort(simMergedClusHitIds.begin(), simMergedClusHitIds.end());
            simMergedClusHitIds.erase(std::unique(simMergedClusHitIds.begin(), simMergedClusHitIds.end()), simMergedClusHitIds.end());

            // -- Get shared hits
            std::vector<uint64_t> sharedHitIds;
            std::sort(recoMergedClusHitIds.begin(), recoMergedClusHitIds.end());
            std::sort(simMergedClusHitIds.begin(), simMergedClusHitIds.end());
            std::set_intersection(recoMergedClusHitIds.begin(),
                                  recoMergedClusHitIds.end(),
                                  simMergedClusHitIds.begin(),
                                  simMergedClusHitIds.end(),
                                  std::back_inserter(sharedHitIds));

            if (sharedHitIds.empty()){
              continue;
              }

            // -- If the sim and reco clusters have common hits, fill the std:vector of sim clusters refs
            if (!sharedHitIds.empty()) {
              simClusterRefs.push_back(simMergedClusterRef);

              LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                  << "RecoToSim --> Found " << sharedHitIds.size() << " shared hits";
              LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                  << "E_recoClus = " << recoMergedClus.energy() << "   E_simClus = " << simMergedClus.simEnergy()
                  << "   E_recoClus/E_simClus = " << recoMergedClus.energy() * 0.001 / simMergedClus.simEnergy();
              LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                  << "(t_recoClus-t_simClus)/sigma_t = "
                  << std::abs((recoMergedClus.time() - simMergedClus.simTime()) / recoMergedClus.timeError());
            }
          }  //end loop over simclusters associated with this detId

        }  //end loop over detIds in reco merged cluster

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
  std::array<edm::Handle<FTLMergedClusterCollection>, 2> inputRecoMergedClusH{{btlRecoClusH, etlRecoClusH}};

  // -- loop over MtdSimMergedClusters
  for (auto simMergedClusIt = simMergedClusters.begin(); simMergedClusIt != simMergedClusters.end();
       simMergedClusIt++) {
    const auto& simMergedClus = *simMergedClusIt;

    
    std::vector<uint64_t> simMergedClusHitIds;
            for (const auto& simLayerClus : simMergedClus.clusters()) {
              for (const auto& hit : simLayerClus->detIds_and_rows()) {
                // Force the hit to map to the macroscopic module ID
                uint32_t modId = geomTools_.sensorModuleId(hit.first);
                
                // Compress the row and column
                uint8_t rowcol = static_cast<uint8_t>((std::clamp(static_cast<int>(hit.second.first), 0, 15) << 4) | 
                                                       std::clamp(static_cast<int>(hit.second.second), 0, 15));
                
                // Construct the 64-bit ID matching the Reco format
                simMergedClusHitIds.push_back((static_cast<uint64_t>(modId) << 8) | static_cast<uint64_t>(rowcol));
              }
            }
            
            // Ensure the generated list is sorted for std::set_intersection
            std::sort(simMergedClusHitIds.begin(), simMergedClusHitIds.end());
            simMergedClusHitIds.erase(std::unique(simMergedClusHitIds.begin(), simMergedClusHitIds.end()), simMergedClusHitIds.end());

    std::vector<DetId> simMergedClusDetIds = simMergedClus.detIds();
    for (size_t i = 0; i < simMergedClusDetIds.size(); ++i) {
      simMergedClusDetIds[i] = geomTools_.sensorModuleId(simMergedClusDetIds[i]);
    }
    std::vector<FTLMergedClusterRef> matchedRecoMergedClusterRefs;
    // loop over reco merged clusters
    for (auto const& recoMergedClusH : inputRecoMergedClusH) {
      for (const auto& detSet : *recoMergedClusH) {
        for (const auto& recoMergedClus : detSet) {
          FTLMergedClusterRef recoMergedClusterRef = edmNew::makeRefTo(recoMergedClusH, &recoMergedClus);
          std::vector<DetId> recodetIds = recoMergedClus.clusterIds();
          std::vector<DetId> shareddetIds;
          std::sort(simMergedClusDetIds.begin(), simMergedClusDetIds.end());
          std::sort(recodetIds.begin(), recodetIds.end());
          std::set_intersection(simMergedClusDetIds.begin(),
                                simMergedClusDetIds.end(),
                                recodetIds.begin(),
                                recodetIds.end(),
                                std::back_inserter(shareddetIds));
          if (shareddetIds.empty())
            continue;  //no shared detIds between sim and reco merged clusters, skip to next reco merged cluster
          else {
            //comput the unique ids of the hits in the reco merged cluster
            std::vector<uint64_t> recoMergedClusHitIds(recoMergedClus.size());
            for (size_t ihit = 0; ihit < recoMergedClus.size(); ++ihit) {
              uint64_t uniqueId = recoMergedClus.hUniqueId(ihit);
              recoMergedClusHitIds[ihit] = uniqueId;
            }

            std::vector<uint64_t> sharedHitIds;
            std::sort(simMergedClusHitIds.begin(), simMergedClusHitIds.end());
            std::sort(recoMergedClusHitIds.begin(), recoMergedClusHitIds.end());
          
            std::set_intersection(simMergedClusHitIds.begin(),
                                  simMergedClusHitIds.end(),
                                  recoMergedClusHitIds.begin(),
                                  recoMergedClusHitIds.end(),
                                  std::back_inserter(sharedHitIds));
            //if no shared hits between sim and reco merged clusters, skip to next reco merged cluster
            if (sharedHitIds.empty())
              continue;
            else {  //if there is an intersection, add the reco merged cluster ref to the vector of matched reco merged clusters
              matchedRecoMergedClusterRefs.push_back(recoMergedClusterRef);
              LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                  << "SimToReco --> Found " << sharedHitIds.size() << " shared hits";
              LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                  << "E_recoClus = " << recoMergedClus.energy() << "   E_simClus = " << simMergedClus.simEnergy()
                  << "   E_recoClus/E_simClus = " << recoMergedClus.energy() * 0.001 / simMergedClus.simEnergy();
              LogDebug("MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl")
                  << "(t_recoClus-t_simClus)/sigma_t = "
                  << std::abs((recoMergedClus.time() - simMergedClus.simTime()) / recoMergedClus.timeError());
            }
          }
        }
      }
    }  //end loop over reco merged clusters

    // Remove duplicates from recoMergedClusterRefs
    std::sort(matchedRecoMergedClusterRefs.begin(), matchedRecoMergedClusterRefs.end());
    matchedRecoMergedClusterRefs.erase(
        std::unique(matchedRecoMergedClusterRefs.begin(), matchedRecoMergedClusterRefs.end()),
        matchedRecoMergedClusterRefs.end());

    edm::Ref<MtdSimMergedClusterCollection> simMergedClusterRef =
        edm::Ref<MtdSimMergedClusterCollection>(simMergedClusH, simMergedClusIt - simMergedClusters.begin());

    outputCollection.emplace_back(simMergedClusterRef, matchedRecoMergedClusterRefs);

  }  // -- end loop over sim merged clusters

  outputCollection.post_insert();
  return outputCollection;
}
