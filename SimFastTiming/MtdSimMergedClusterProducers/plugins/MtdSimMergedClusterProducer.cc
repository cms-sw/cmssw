// Enable debug logging
#define EDM_ML_DEBUG
#include <chrono>
#include <FWCore/Framework/interface/one/EDProducer.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

// Logging
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"

#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedClusterFwd.h"

// MTD truth association maps
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToRecoClusterAssociationMap.h"

// Geometry and topology
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

// DetId
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <memory>
#include <set>

void traverseDecayTree(const edm::Ref<TrackingParticleCollection>& tpRef,
                       std::set<edm::Ref<TrackingParticleCollection>>& visited,
                       const std::function<void(const edm::Ref<TrackingParticleCollection>&)>& action) {
  // stop condition
  if (visited.count(tpRef))
    return;

  // keep track of visited particles
  visited.insert(tpRef);

  // Perform the user-defined action (e.g. fill mergedcluster)
  action(tpRef);

  const auto& decayVtxs = tpRef->decayVertices();
  if (!decayVtxs.empty()) {
    // iterate using begin, end explicitly
    for (auto it = decayVtxs.begin(); it != decayVtxs.end(); ++it) {
      const auto& decayVtx = *it;
      for (const auto& daughterRef : decayVtx->daughterTracks()) {
        traverseDecayTree(daughterRef, visited, action);
      }
    }
  }
}

class MtdSimMergedClusterProducer : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit MtdSimMergedClusterProducer(const edm::ParameterSet&);
  ~MtdSimMergedClusterProducer() override = default;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticlesToken_;
  edm::EDGetTokenT<reco::TPToSimCollectionMtd> tpToSimClusMapToken_;
  edm::EDGetTokenT<reco::SimToTPCollectionMtd> simClusToTPMapToken_;
  edm::EDGetTokenT<MtdSimLayerClusterCollection> mtdSimLayerClustersToken_;
  double minEnergy_;

  const edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
};

MtdSimMergedClusterProducer::MtdSimMergedClusterProducer(const edm::ParameterSet& iConfig)
    : trackingParticlesToken_(
          consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles"))),
      tpToSimClusMapToken_(
          consumes<reco::TPToSimCollectionMtd>(iConfig.getParameter<edm::InputTag>("tp2SimAssociationMap"))),
      simClusToTPMapToken_(
          consumes<reco::SimToTPCollectionMtd>(iConfig.getParameter<edm::InputTag>("tp2SimAssociationMap"))),
      mtdSimLayerClustersToken_(
          consumes<MtdSimLayerClusterCollection>(iConfig.getParameter<edm::InputTag>("mtdSimLayerClusters"))),
      minEnergy_(iConfig.getParameter<double>("minClusterEnergy")),
      mtdtopoToken_(esConsumes<MTDTopology, MTDTopologyRcd>()),
      mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()) {
  produces<MtdSimMergedClusterCollection>();
}

void MtdSimMergedClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get topology for navigation
  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();
  auto const& geom = iSetup.getData(mtdgeoToken_);

  static constexpr uint32_t halfTrayBTL_SMidx = MTDTopology::BTLLayout::nBTLeta_ / 2;

  // Create output collection (MtdSimMergedCluster)
  auto outputClusters = std::make_unique<MtdSimMergedClusterCollection>();

  // Retrieve collections
  edm::Handle<TrackingParticleCollection> trackingParticles;
  iEvent.getByToken(trackingParticlesToken_, trackingParticles);

  edm::Handle<reco::TPToSimCollectionMtd> tpToSimClusMap;
  iEvent.getByToken(tpToSimClusMapToken_, tpToSimClusMap);

  edm::Handle<reco::SimToTPCollectionMtd> simClusToTPMap;
  iEvent.getByToken(simClusToTPMapToken_, simClusToTPMap);

  edm::Handle<MtdSimLayerClusterCollection> simLClusters;
  iEvent.getByToken(mtdSimLayerClustersToken_, simLClusters);

  struct TPRefHash {
    std::size_t operator()(const TrackingParticleRef& ref) const {
      // Extract the unique index of the particle in the collection and hash it
      return std::hash<size_t>()(ref.key());
    }
  };

  std::unordered_map<TrackingParticleRef, TrackingParticleRef, TPRefHash> ancestorCache;

  // the map is filled on-demand
  auto getEarliestAncestor = [&ancestorCache](TrackingParticleRef tp) -> TrackingParticleRef {
    //already seen this TP, return ancestor instantly
    if (ancestorCache.count(tp)) {
      return ancestorCache[tp];
    }

    TrackingParticleRef current = tp;

    // Use unordered_set to avoid heap-allocation thrashing for the infinite loop check
    std::unordered_set<TrackingParticleRef, TPRefHash> visited;

    while (true) {
      if (visited.count(current)) {
        break;
      }
      visited.insert(current);

      const auto& parentVertices = current->parentVertex();
      if (parentVertices.isNull() || !parentVertices.isAvailable()) {
        break;
      }
      const auto& parentTracks = parentVertices->sourceTracks();
      if (parentTracks.empty()) {
        break;
      }

      current = parentTracks[0];

      // SHORTCUT: If our parent is already in the cache, we don't need to traverse the rest of the tree
      if (ancestorCache.count(current)) {
        current = ancestorCache[current];
        break;
      }
    }

    //update the cache
    ancestorCache[tp] = current;
    return current;
  };

  LogDebug("MtdSimMergedClusterProducer") << "Total TrackingParticles: " << trackingParticles->size();

  // reserve memory for output collection
  // (worst case scenario: all TrackingParticles are primary)
  outputClusters->reserve(trackingParticles->size());

  // Create cluster map for fast lookup (can have multiple clusters per DetId)
  std::vector<const MtdSimLayerCluster*> allsimLClusters;
  std::vector<const MtdSimLayerCluster*> allsimETLLClusters;
  for (const auto& cluster : *simLClusters) {
    if (!cluster.detIds_and_rows().empty()) {
      if (MTDDetId(cluster.detIds_and_rows()[0].first).mtdSubDetector() == MTDDetId::ETL) {
        allsimETLLClusters.push_back(&cluster);
      } else if (MTDDetId(cluster.detIds_and_rows()[0].first).mtdSubDetector() == MTDDetId::BTL) {
        allsimLClusters.push_back(&cluster);
      }
    }
  }
  BTLDetId::CrysLayout crysLayout = MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode());
  // Sort simLClusters collection
  std::sort(allsimLClusters.begin(),
            allsimLClusters.end(),
            [&topology, crysLayout](const MtdSimLayerCluster* a, const MtdSimLayerCluster* b) {
              const auto& detIdsA = a->detIds_and_rows();
              const auto& detIdsB = b->detIds_and_rows();

              auto [iphiA, ietaA] = topology->btlIndex(BTLDetId::rawGeoId(detIdsA[0].first, crysLayout));
              auto [iphiB, ietaB] = topology->btlIndex(BTLDetId::rawGeoId(detIdsB[0].first, crysLayout));

              if (iphiA != iphiB)
                return iphiA < iphiB;
              if (ietaA != ietaB)
                return ietaA < ietaB;

              // Get min and max columns for both clusters
              auto [minA, maxA] = std::minmax_element(detIdsA.begin(), detIdsA.end(), [](auto const& x, auto const& y) {
                return x.second.second < y.second.second;
              });
              auto [minB, maxB] = std::minmax_element(detIdsB.begin(), detIdsB.end(), [](auto const& x, auto const& y) {
                return x.second.second < y.second.second;
              });

              int lowest_icol_A = minA->second.second;
              int highest_icol_A = maxA->second.second;
              int lowest_icol_B = minB->second.second;
              int highest_icol_B = maxB->second.second;
              int widthA = highest_icol_A - lowest_icol_A;
              int widthB = highest_icol_B - lowest_icol_B;
              if (widthA != widthB)
                return widthA > widthB;  // larger cluster first

              if (ietaA <= halfTrayBTL_SMidx) {  // put "leftmost" (lower z) cluster first
                return highest_icol_A > highest_icol_B;
              } else {
                return lowest_icol_A < lowest_icol_B;
              }
            });

  // Now, construct cluster map from ordered collection
  std::unordered_map<uint32_t, std::vector<const MtdSimLayerCluster*>> clusterMap;

  for (const auto* cluster : allsimLClusters) {
    if (cluster->energy() >= minEnergy_) {
      // retrieve GEOGRAPHICAL id -> rawId
      clusterMap[BTLDetId::rawGeoId(cluster->detIds_and_rows()[0].first, crysLayout)].push_back(cluster);    
    }
  }

  LogDebug("MtdSimMergedClusterProducer")
      << "Found " << clusterMap.size() << " MTD SimLayerClusters above energy threshold";

  // ------------------------------------------------------
  // Using TOPOLOGICAL + HISTORICAL clustering algorithm

  std::unordered_set<const MtdSimLayerCluster*> processedClusters;
  std::vector<const MtdSimLayerCluster*> mergedClusterClusters;
  // Process clusters with full merging logic
  for (const auto* clusterPointer : allsimLClusters) {
    const auto& cluster = *clusterPointer;

    if (cluster.energy() < minEnergy_ || processedClusters.count(&cluster))
      continue;

    // Start with current cluster
    mergedClusterClusters.clear();
    mergedClusterClusters.push_back(&cluster);
    processedClusters.insert(&cluster);

    // Check for edge hits in current cluster
    bool edgeHitIn0 = false;
    bool edgeHitIn15 = false;

    // iterate over detIds_and_rows:
    LogDebug("MtdSimMergedClusterProducer") << "Iterating over " << cluster.detIds_and_rows().size() << " cluster hits";
    auto detids_and_rows_ = cluster.detIds_and_rows();
    auto [lowest_col_, highest_col_] =
        std::minmax_element(detids_and_rows_.begin(), detids_and_rows_.end(), [](auto const& x, auto const& y) {
          return x.second.second < y.second.second;
        });

    uint8_t lowest_col = lowest_col_->second.second;
    uint8_t highest_col = highest_col_->second.second;

    if (lowest_col == 0) {
      edgeHitIn0 = true;
    } else if (highest_col == 15) {
      edgeHitIn15 = true;
    }

    LogDebug("MtdSimMergedClusterProducer") << "  Edge hits: col0=" << edgeHitIn0 << ", col15=" << edgeHitIn15;

    uint32_t thisCluIdRaw = BTLDetId::rawGeoId(cluster.detIds_and_rows()[0].first, crysLayout);

    // Get topology indices - use geographicalId (module-level) with crystal layout
    std::pair<uint32_t, uint32_t> indices = topology->btlIndex(thisCluIdRaw);
    uint32_t iphi = indices.first;
    uint32_t ieta = indices.second;
    LogDebug("MtdSimMergedClusterProducer") << "  BTL indices: iphi=" << iphi << ", ieta=" << ieta;

    bool hasEdgeHitCurrent = false;
    bool positiveZ = ieta > halfTrayBTL_SMidx;
    if (((!positiveZ) && edgeHitIn0) ||
        (positiveZ && edgeHitIn15)) {  //check if there an edge hit on the right side (higher z)
      hasEdgeHitCurrent = true;
    }
    LogDebug("MtdSimMergedClusterProducer") << "  hasEdgeHitCurrent = " << hasEdgeHitCurrent;

    // MERGING IN SAME SM + NEXT SM IN ETA DIRECTION
    std::vector<int> etaOffsets = {1, 0};
    for (int etaOffset : etaOffsets) {
      if ((hasEdgeHitCurrent) ||
          etaOffset ==
              0) {  // if there is an edge hit, we can merge in eta direction +1, otherwise only same-SM merging.
        if ((ieta == halfTrayBTL_SMidx) && etaOffset == 1) {
          continue;  // skip merging across the eta=0 gap
        }
        uint32_t adjDetIdRaw;
        if (etaOffset == 1) {
          adjDetIdRaw = topology->btlidFromIndex(iphi, ieta + etaOffset);
          LogDebug("MtdSimMergedClusterProducer")
              << "    Checking adjacent detId at index (" << iphi << ", " << ieta + etaOffset << "): " << adjDetIdRaw;
          if (adjDetIdRaw == 0)  //at the end of the tray in eta direction
            continue;
        } else {  //etaOffset == 0, same-SM merging
          adjDetIdRaw = thisCluIdRaw;
        }

        auto it = clusterMap.find(adjDetIdRaw);
        if (it == clusterMap.end()) {
          LogDebug("MtdSimMergedClusterProducer") << "No cluster found at this detId";
          continue;
        }

        // Iterate over all clusters at this DetId
        for (const MtdSimLayerCluster* adjCluster : it->second) {
          if (adjCluster == &cluster) {  //skip same cluster in SM
            continue;
          }
          if (processedClusters.count(adjCluster)) {  //skip already processed clusters
            LogDebug("MtdSimMergedClusterProducer") << "Cluster found but already processed";
            continue;
          }

          LogDebug("MtdSimMergedClusterProducer") << "Found cluster at " << adjDetIdRaw;

          // Check for opposite edge hit
          bool hasOppositeEdgeHit = false;
          auto adj_detids_and_rows_ = adjCluster->detIds_and_rows();

          auto [adj_lowest_col_, adj_highest_col_] = std::minmax_element(
              adj_detids_and_rows_.begin(), adj_detids_and_rows_.end(), [](auto const& x, auto const& y) {
                return x.second.second < y.second.second;
              });

          uint8_t adj_lowest_col = adj_lowest_col_->second.second;
          uint8_t adj_highest_col = adj_highest_col_->second.second;

          if ((edgeHitIn15 && (adj_lowest_col == 0) && positiveZ) ||
              (edgeHitIn0 && (adj_highest_col == 15) &&
               !positiveZ)) {  //check if there is an edge hit in the neighbouring cluster
            hasOppositeEdgeHit = true;
          }
          bool areClustersOverlapping = false;
          if (etaOffset == 0) {
            int clu_len = highest_col - lowest_col + 1;

            bool isLeftEdgeOverlapping =
                (abs(adj_lowest_col - lowest_col) <= clu_len) &&
                (abs(highest_col - adj_lowest_col) <= clu_len);  // equality includes adjacent clusters
            bool isRightEdgeOverlapping =
                (abs(adj_highest_col - highest_col) <= clu_len) && (abs(lowest_col - adj_highest_col) <= clu_len);
            areClustersOverlapping = isLeftEdgeOverlapping || isRightEdgeOverlapping;
          }

          if ((hasEdgeHitCurrent && hasOppositeEdgeHit) || (areClustersOverlapping)) {
            // check for common ancestor
            bool hasCommonAncestor = false;
            bool areBothDirect = (mergedClusterClusters[0]->hitProdType() == 0) && (adjCluster->hitProdType() == 0);
            bool areBothDirectfromSameTP = false;

            // if both clusters are direct, check if they come from the same TP, otherwise keep them separate, even if they have a common ancestor
            if (areBothDirect) {
              const auto& simLayerClusters1 =
                  simClusToTPMap->find(MtdSimLayerClusterRef(simLClusters, &cluster - &(*simLClusters->begin())));
              const auto& simLayerClusters2 =
                  simClusToTPMap->find(MtdSimLayerClusterRef(simLClusters, adjCluster - &(*simLClusters->begin())));
              if (simLayerClusters1 != simClusToTPMap->end() && simLayerClusters2 != simClusToTPMap->end()) {
                for (const auto& tpRef1 : simLayerClusters1->val) {
                  for (const auto& tpRef2 : simLayerClusters2->val) {
                    if (tpRef1 == tpRef2) {
                      areBothDirectfromSameTP = true;
                    }
                  }
                }
              }
            } else {  // if at least one of the clusters is not direct, check for common ancestor and merge if they share one, regardless of whether they come from the same TP or not
              const auto& simLayerClusters1 =
                  simClusToTPMap->find(MtdSimLayerClusterRef(simLClusters, &cluster - &(*simLClusters->begin())));
              const auto& simLayerClusters2 =
                  simClusToTPMap->find(MtdSimLayerClusterRef(simLClusters, adjCluster - &(*simLClusters->begin())));
              if (simLayerClusters1 != simClusToTPMap->end() && simLayerClusters2 != simClusToTPMap->end()) {
                for (const auto& tpRef1 : simLayerClusters1->val) {
                  for (const auto& tpRef2 : simLayerClusters2->val) {
                    if (getEarliestAncestor(tpRef1) == getEarliestAncestor(tpRef2)) {
                      hasCommonAncestor = true;
                      break;
                    }
                  }
                  if (hasCommonAncestor)
                    break;
                }
              }
            }
            if (hasCommonAncestor || areBothDirectfromSameTP) {
              LogDebug("MtdSimMergedClusterProducer")
                  << "  -> MERGING: cluster in " << thisCluIdRaw << " with " << adjDetIdRaw;
              bool isBackscatterMergedcluster = mergedClusterClusters[0]->hitProdType() == 3;
              bool areBothBackscatter = isBackscatterMergedcluster && (adjCluster->hitProdType() == 3);
              bool areBothNotBackscatter = !isBackscatterMergedcluster && (adjCluster->hitProdType() != 3);
              if (areBothBackscatter || (areBothNotBackscatter && !areBothDirect) || areBothDirectfromSameTP) {
                mergedClusterClusters.push_back(adjCluster);
                processedClusters.insert(adjCluster);

              } else {
                LogDebug("MtdSimMergedClusterProducer")
                    << "    Not merging: different hitProdType and not both direct from the same TP";
              }

            } else {
              LogDebug("MtdSimMergedClusterProducer") << "    Not merging: no common ancestor found";
            }
          }
        }
      } else {
        LogDebug("MtdSimMergedClusterProducer")
            << "  No edge hit, invalid indices or same module check: not attempting eta merging";
        LogDebug("MtdSimMergedClusterProducer")
            << "    hasEdgeHitCurrent = " << hasEdgeHitCurrent << ", iphi = " << iphi << " (is max? "
            << (iphi == std::numeric_limits<uint32_t>::max()) << "), ieta = " << ieta << " (is max? "
            << (ieta == std::numeric_limits<uint32_t>::max()) << ")";
      }
    }

    // Create MergedCluster from merged clusters
    MtdSimMergedCluster simMergedCluster;

    // for each cluster, find associated TPs and add to mergedcluster using Sim to TP map
    for (const auto& simLayerCluster : mergedClusterClusters) {
      // create simLC reference by finding the index in the original collection
      size_t clusterIndex = simLayerCluster - &(*simLClusters->begin());
      MtdSimLayerClusterRef simLayerClusterRef(simLClusters, clusterIndex);
      const auto& TPs = simClusToTPMap->find(simLayerClusterRef);
      if (TPs != simClusToTPMap->end()) {
        for (const auto& tpRef : TPs->val) {
          simMergedCluster.addCluster(simLayerClusterRef, tpRef);
        }
      } else {
        LogDebug("MtdSimMergedClusterProducer") << "No TP associated to cluster index " << clusterIndex;
        // add null reference
        simMergedCluster.addCluster(simLayerClusterRef, TrackingParticleRef());
      }
    }
    if (mergedClusterClusters.size() == 1) {
      // if only one cluster, take position and time from it
      simMergedCluster.setSimPos(mergedClusterClusters[0]->simLCPos());
    } else {
      // --- Calculate energy-weighted position ---
      double weightedGlobalX = 0;
      double weightedGlobalY = 0;
      double weightedGlobalZ = 0;
      float totalEnergy = 0;

      for (const auto& simLCptr : mergedClusterClusters) {
        const MtdSimLayerCluster& simLC = *simLCptr;
        float energy = simLC.simLCEnergy();
        totalEnergy += energy;

        // Use the first hit's DetId for geometry lookup
        if (!simLC.detIds_and_rows().empty()) {
          DetId detId = simLC.detIds_and_rows()[0].first;
          const GeomDet* det = geom.idToDetUnit(detId);
          if (det) {
            const GlobalPoint& gp = det->surface().toGlobal(simLC.simLCPos());
            weightedGlobalX += static_cast<double>(energy) * gp.x();
            weightedGlobalY += static_cast<double>(energy) * gp.y();
            weightedGlobalZ += static_cast<double>(energy) * gp.z();
          }
        }
      }

      GlobalPoint avgGlobal(0., 0., 0.);
      if (totalEnergy > 0) {
        avgGlobal =
            GlobalPoint(weightedGlobalX / totalEnergy, weightedGlobalY / totalEnergy, weightedGlobalZ / totalEnergy);
      }

      // Convert back to local coordinates of the seed cluster
      if (!mergedClusterClusters.empty()) {
        uint32_t seedRawId = mergedClusterClusters.front()->detIds_and_rows()[0].first;
        DetId seedGeoId = DetId(BTLDetId::rawGeoId(seedRawId, crysLayout));
        const GeomDet* seedDet = geom.idToDetUnit(seedGeoId);

        if (seedDet) {
          LocalPoint lp = seedDet->surface().toLocal(avgGlobal);
          simMergedCluster.setSimPos(lp);
        } else {
          edm::LogWarning("MtdSimMergedClusterProducer") << "Could not find seed detector for position calculation";
          simMergedCluster.setSimPos(mergedClusterClusters[0]->simLCPos());
        }
      }
    }
    // --- End position calculation ---

    outputClusters->push_back(simMergedCluster);
#ifdef EDM_ML_DEBUG
    LogDebug("MtdSimMergedClusterProducer")
        << "Created MergedCluster from " << mergedClusterClusters.size()
        << " clusters: E=" << simMergedCluster.simEnergy() << " MeV, t=" << simMergedCluster.simTime() << " ns";
#endif
  }
  // For ETL: copy paste of original MtdSimLayerClusters
  for (const auto* clusterPointer : allsimETLLClusters) {
    MtdSimMergedCluster simMergedCluster;
    // create simLC reference by finding the index in the original collection
    size_t clusterIndex = clusterPointer - &(*simLClusters->begin());
    MtdSimLayerClusterRef simLayerClusterRef(simLClusters, clusterIndex);
    const auto& TPs = simClusToTPMap->find(simLayerClusterRef);
    if (TPs != simClusToTPMap->end()) {
      for (const auto& tpRef : TPs->val) {
        simMergedCluster.addCluster(simLayerClusterRef, tpRef);
      }
    } else {
      LogDebug("MtdSimMergedClusterProducer") << "No TP associated to cluster index " << clusterIndex;
      // add null reference
      simMergedCluster.addCluster(simLayerClusterRef, TrackingParticleRef());
    }
    simMergedCluster.setSimPos(clusterPointer->simLCPos());
    outputClusters->push_back(simMergedCluster);
    LogDebug("MtdSimMergedClusterProducer")
        << "Created ETL MergedCluster from an ETL cluster"
        << " E=" << simMergedCluster.simEnergy() << " MeV, t=" << simMergedCluster.simTime() << " ns";
  }

  iEvent.put(std::move(outputClusters));
}

DEFINE_FWK_MODULE(MtdSimMergedClusterProducer);
