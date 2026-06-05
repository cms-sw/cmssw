// Enable debug logging
#define EDM_ML_DEBUG

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
  
  static constexpr uint32_t halfTrayBTL_SMidx = MTDTopology::BTLLayout::nBTLeta_/2;
  
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

  // BUILD ANCESTOR MAP: to check if 2 TrackingParticles have the same root ancestor
  std::map<TrackingParticleRef, TrackingParticleRef> ancestorMap;
  for (size_t i = 0; i < trackingParticles->size(); ++i) {
    TrackingParticleRef tp(trackingParticles, i);

    // Find earliest ancestor for this TP
    TrackingParticleRef current = tp;
    std::set<TrackingParticleRef> visited;

    while (true) {
      // Prevent infinite loops
      if (visited.count(current)) {
        break;
      }
      visited.insert(current);

      // Check if this particle has parent vertices
      const auto& parentVertices = current->parentVertex();
      if (parentVertices.isNull() || !parentVertices.isAvailable()) {
        // No parent vertex - this is the root ancestor
        break;
      }
      // Get the parent tracks from the parent vertex
      const auto& parentTracks = parentVertices->sourceTracks();
      if (parentTracks.empty()) {
        // No parent tracks - this is the root ancestor
        break;
      }
      // Move to the first parent
      current = parentTracks[0];
    }
    // Store the mapping: TP -> earliest ancestor
    ancestorMap[tp] = current;
  }

  // ANCESTOR MAP SANITY CHECK: how many unique ancestors are there?
  std::set<TrackingParticleRef> uniqueAncestors;
  for (const auto& pair : ancestorMap) {
    uniqueAncestors.insert(pair.second);
  }
  LogDebug("MtdSimMergedClusterProducer") << "Total TrackingParticles: " << trackingParticles->size()
                                          << ", Unique primary ancestors: " << uniqueAncestors.size();

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

  // Sort simLClusters collection
  std::sort(allsimLClusters.begin(),
            allsimLClusters.end(),
            [&topology](const MtdSimLayerCluster* a, const MtdSimLayerCluster* b) {
              const auto& detIdsA = a->detIds_and_rows();
              const auto& detIdsB = b->detIds_and_rows();

              BTLDetId idA(detIdsA[0].first);
              BTLDetId idB(detIdsB[0].first);

              auto [iphiA, ietaA] =
                  topology->btlIndex(idA.geographicalId(BTLDetId::CrysLayout::v4).rawId());  //uint32_t
              auto [iphiB, ietaB] = topology->btlIndex(idB.geographicalId(BTLDetId::CrysLayout::v4).rawId());

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

              if (ietaA < 49) {  // put "leftmost" (lower z) cluster first
                return lowest_icol_A > lowest_icol_B;
              } else {
                return lowest_icol_A < lowest_icol_B;
              }
            });

  // Now, construct cluster map from ordered collection
  std::map<BTLDetId, std::vector<const MtdSimLayerCluster*>> clusterMap;

  for (const auto* cluster : allsimLClusters) {
    if (cluster->energy() >= minEnergy_) {
      // retrieve detId from first hit
      BTLDetId detId = cluster->detIds_and_rows()[0].first;
      // retrieve GEOGRAPHICAL id
      BTLDetId geoDetId = detId.geographicalId(BTLDetId::CrysLayout::v4);

      clusterMap[geoDetId].push_back(cluster);
    }
  }

  LogDebug("MtdSimMergedClusterProducer")
      << "Found " << clusterMap.size() << " MTD SimLayerClusters above energy threshold";

  // ------------------------------------------------------
  // Using TOPOLOGICAL + HISTORICAL clustering algorithm

  std::set<const MtdSimLayerCluster*> processedClusters;

  // Process clusters with full merging logic
  for (const auto* clusterPointer : allsimLClusters) {
    const auto& cluster = *clusterPointer;

    if (cluster.energy() < minEnergy_ || processedClusters.count(&cluster))
      continue;

    // // TEMPORARY: forget about ETL hits
    if (!cluster.detIds_and_rows().empty() &&
        MTDDetId(cluster.detIds_and_rows()[0].first).mtdSubDetector() == MTDDetId::ETL)
      continue;

    BTLDetId cluId(cluster.detIds_and_rows()[0].first);

    // Start with current cluster
    std::vector<const MtdSimLayerCluster*> mergedClusterClusters = {&cluster};
    processedClusters.insert(&cluster);

    // Check for edge hits in current cluster
    bool edgeHitIn0 = false;
    bool edgeHitIn15 = false;
    int leftmost_col = 20;
    int rightmost_col = -1;

    // iterate over detIds_and_rows:
    LogDebug("MtdSimMergedClusterProducer") << "Iterating over " << cluster.detIds_and_rows().size() << " cluster hits";
    for (auto const& detId_row_col : cluster.detIds_and_rows()) {
      int row = detId_row_col.second.first;
      int col = detId_row_col.second.second;

      LogDebug("MtdSimMergedClusterProducer") << "  Cluster hit at row " << row << ", col " << col;

      if (col == 0) {
        edgeHitIn0 = true;
      } else if (col == 15) {
        edgeHitIn15 = true;
      }

      if (col < leftmost_col)
        leftmost_col = col;
      if (col > rightmost_col)
        rightmost_col = col;
    }

    LogDebug("MtdSimMergedClusterProducer") << "  Edge hits: col0=" << edgeHitIn0 << ", col15=" << edgeHitIn15;

    // Get topology indices - use geographicalId (module-level) with crystal layout
    std::pair<uint32_t, uint32_t> indices = topology->btlIndex(cluId.geographicalId(BTLDetId::CrysLayout::v4).rawId());
    uint32_t iphi = indices.first;
    uint32_t ieta = indices.second;
    LogDebug("MtdSimMergedClusterProducer") << "  BTL indices: iphi=" << iphi << ", ieta=" << ieta;

    bool hasEdgeHitCurrent = false;
    if (((ieta < halfTrayBTL_SMidx) && edgeHitIn0) ||
        ((ieta > halfTrayBTL_SMidx) && edgeHitIn15)) {  //check if there an edge hit on the right side (higher z)
      hasEdgeHitCurrent = true;
    }
    LogDebug("MtdSimMergedClusterProducer") << "  hasEdgeHitCurrent = " << hasEdgeHitCurrent;

    // ETA DIRECTION MERGING
    std::vector<int> etaOffsets = {1, 0};
    for (int etaOffset : etaOffsets) {
      if ((hasEdgeHitCurrent && iphi != std::numeric_limits<uint32_t>::max() &&
           ieta != std::numeric_limits<uint32_t>::max()) ||
          etaOffset == 0) {
        if ((ieta == halfTrayBTL_SMidx) && etaOffset == 1) {
          continue;  // skip merging across the eta=0 gap
        }
        LogDebug("MtdSimMergedClusterProducer") << "  Attempting eta-direction merging...";
        uint32_t adjDetIdRaw = topology->btlidFromIndex(iphi, ieta + etaOffset);
        LogDebug("MtdSimMergedClusterProducer")
            << "    Checking adjacent detId at index (" << iphi << ", " << ieta + etaOffset << "): " << adjDetIdRaw;
        if (adjDetIdRaw == 0)
          continue;

        BTLDetId adjDetId(adjDetIdRaw);
        auto it = clusterMap.find(adjDetId);
        if (it == clusterMap.end()) {
          LogDebug("MtdSimMergedClusterProducer") << "No cluster found at this detId";
          continue;
        }

        // Iterate over all clusters at this DetId
        for (const MtdSimLayerCluster* adjCluster : it->second) {
          if (processedClusters.count(adjCluster)) {
            LogDebug("MtdSimMergedClusterProducer") << "Cluster found but already processed";
            continue;
          }

          LogDebug("MtdSimMergedClusterProducer") << "Found eta neighbor at " << adjDetId.rawId();

          // Check for opposite edge hit
          bool hasOppositeEdgeHit = false;
          int adj_leftmost_col = 20;
          int adj_rightmost_col = -1;

          for (auto const& detId_row_col : adjCluster->detIds_and_rows()) {
            int row = detId_row_col.second.first;
            int col = detId_row_col.second.second;

            LogDebug("MtdSimMergedClusterProducer") << " adjacent cluster hit at row " << row << ", col " << col;

            if ((edgeHitIn15 && col == 0 && ieta > halfTrayBTL_SMidx) ||
                (edgeHitIn0 && col == 15 && ieta < halfTrayBTL_SMidx)) {  //check if there is an edge hit in the neighbouring cluster
              hasOppositeEdgeHit = true;
            }

            if (col < adj_leftmost_col)
              adj_leftmost_col = col;
            if (col > adj_rightmost_col)
              adj_rightmost_col = col;
          }

          int clu_len = rightmost_col - leftmost_col + 1;

          bool isLeftEdgeOverlapping =
              (abs(adj_leftmost_col - leftmost_col) <= clu_len) &&
              (abs(rightmost_col - adj_leftmost_col) <= clu_len);  // equality includes adjacent clusters
          bool isRightEdgeOverlapping =
              (abs(adj_rightmost_col - rightmost_col) <= clu_len) && (abs(leftmost_col - adj_rightmost_col) <= clu_len);

          bool areClustersOverlapping = isLeftEdgeOverlapping || isRightEdgeOverlapping;

          if (hasOppositeEdgeHit || (areClustersOverlapping && etaOffset == 0)) {
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
                    if (ancestorMap[tpRef1] == ancestorMap[tpRef2]) {
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
                  << "  -> MERGING ETA neighbor: " << cluId.rawId() << " with " << adjDetId.rawId();
              int iphi_adj, ieta_adj;
              std::tie(iphi_adj, ieta_adj) =
                  topology->btlIndex(adjDetId.geographicalId(BTLDetId::CrysLayout::v4).rawId());
              bool isBackscatterMergedcluster = mergedClusterClusters[0]->hitProdType() == 3;
              bool areBothBackscatter = isBackscatterMergedcluster && (adjCluster->hitProdType() == 3);
              bool areBothNotBackscatter = !isBackscatterMergedcluster && (adjCluster->hitProdType() != 3);
              if (areBothBackscatter || (areBothNotBackscatter && !areBothDirect) || areBothDirectfromSameTP) {
                mergedClusterClusters.push_back(adjCluster);
                processedClusters.insert(adjCluster);
                
              }
              else {
                LogDebug("MtdSimMergedClusterProducer") << "    Not merging: different hitProdType and not both direct from the same TP"; 
              }
              
            } else {
              int iphi_adj, ieta_adj;
              std::tie(iphi_adj, ieta_adj) =
                  topology->btlIndex(adjDetId.geographicalId(BTLDetId::CrysLayout::v4).rawId());
              
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
    if (mergedClusterClusters.size()==1) {
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
        DetId seedDetId = mergedClusterClusters.front()->detIds_and_rows()[0].first;
        BTLDetId seedBTL(seedDetId);
        DetId seedGeoId = seedBTL.geographicalId(BTLDetId::CrysLayout::v4);

        const GeomDet* seedDet = geom.idToDetUnit(seedGeoId);
        
        if (seedDet) {
          LocalPoint lp = seedDet->surface().toLocal(avgGlobal);
          simMergedCluster.setSimPos(lp);
        }
        else{
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
  }
#endif

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
    outputClusters->push_back(simMergedCluster);
    LogDebug("MtdSimMergedClusterProducer")
        << "Created ETL MergedCluster from an ETL cluster"
        << " E=" << simMergedCluster.simEnergy() << " MeV, t=" << simMergedCluster.simTime() << " ns";
  }

  iEvent.put(std::move(outputClusters));
}

DEFINE_FWK_MODULE(MtdSimMergedClusterProducer);
