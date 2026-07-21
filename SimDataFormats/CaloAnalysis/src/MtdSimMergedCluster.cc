#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <iostream>

MtdSimMergedCluster::MtdSimMergedCluster(const MtdSimLayerClusterRef& clusterRef, const TrackingParticleRef& tpRef) {
  addCluster(clusterRef, tpRef);
}

void MtdSimMergedCluster::addCluster(const MtdSimLayerClusterRef& clusterRef, const TrackingParticleRef& tpRef) {
  clusters_.push_back(clusterRef);

  // check if tpRef is valid
  if (tpRef.isNonnull()) {
    // check if tpRef is already in trackingParticles_
    auto it = std::find(trackingParticles_.begin(), trackingParticles_.end(), tpRef);
    if (it == trackingParticles_.end()) {
      trackingParticles_.push_back(tpRef);
    }
  }

  // Sort clusters by time (earliest first)
  std::vector<MtdSimLayerClusterRef> sortedRefs(clusters_.begin(), clusters_.end());
  std::sort(sortedRefs.begin(), sortedRefs.end(), [](const MtdSimLayerClusterRef& a, const MtdSimLayerClusterRef& b) {
    return a->simLCTime() < b->simLCTime();
  });
  clusters_.clear();
  for (auto& ref : sortedRefs)
    clusters_.push_back(ref);
}

float MtdSimMergedCluster::simTime() const {
  // use energy-weighted average of cluster times:
  float time = 0;
  float totalEnergy = 0;
  for (const auto& clu : clusters_) {
    totalEnergy += clu->simLCEnergy();
    time += clu->simLCTime() * clu->simLCEnergy();
  }
  if (totalEnergy > 0) {
    return time / totalEnergy;
  } else {
    return -999;
  }
}

float MtdSimMergedCluster::simEnergy() const {
  float totalEnergy = 0;
  for (const auto& clu : clusters_) {
    // FIRST IMPLEMENTATION: sum energies of *all* clusters
    totalEnergy += clu->simLCEnergy();
  }
  return totalEnergy;
}

DetId MtdSimMergedCluster::simDetId() const {
  if (clusters_.empty()) {
    return -999;
  } else {
    // FIRST IMPLEMENTATION: take detId of earliest sim LC
    auto detIds_and_rows = (*clusters_.begin())->detIds_and_rows();
    return detIds_and_rows[0].first;
  }
}

std::vector<DetId> MtdSimMergedCluster::detIds() const {
  std::vector<DetId> ids;
  for (const auto& clu : clusters_) {
    const auto& clusterDetIds_and_rows = clu->detIds_and_rows();
   for (const auto& hit : clusterDetIds_and_rows) {
      // hit.first contains the valid 32-bit DetId
      DetId id(hit.first);
      
      // Add it to the vector if it is not already present
      if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
        ids.push_back(id);
      }
    }
  }

  return ids;
}

std::vector<std::pair<float, LocalPoint>> MtdSimMergedCluster::hitTimesAndPositions() const {
  std::vector<std::pair<float, LocalPoint>> hitTimesAndPositions;

  for (const auto& clu : clusters_) {
    const auto& cluster_hit_times = clu->hits_and_times();
    const auto& cluster_hit_positions = clu->hits_and_positions();

    // iterate over the two maps in parallel
    std::transform(cluster_hit_times.begin(),
                   cluster_hit_times.end(),
                   cluster_hit_positions.begin(),
                   std::inserter(hitTimesAndPositions, hitTimesAndPositions.end()),
                   [](const auto& time_pair, const auto& pos_pair) {
                     // time_pair is pair<detId, time>, pos_pair is pair<detId, position>
                     float hitTime = time_pair.second;
                     LocalPoint hitPosition = pos_pair.second;
                     if (time_pair.first != pos_pair.first) {
                       LogDebug("MtdSimMergedCluster") << "Warning: Mismatched detIds in hit times and positions!";
                       return std::make_pair(-1.0f, LocalPoint(-999, -999, -999));
                     } else {
                       return std::make_pair(hitTime, hitPosition);
                     }
                   });
  }

  // before returning, sort by time
  std::sort(
      hitTimesAndPositions.begin(),
      hitTimesAndPositions.end(),
      [](const std::pair<float, LocalPoint>& a, const std::pair<float, LocalPoint>& b) { return a.first < b.first; });

  return hitTimesAndPositions;
}

std::vector<uint64_t> MtdSimMergedCluster::hitUniqueIds()
    const {  //unique ids, computed in the same way as in FTLMergedCluster, for all hits in the merged cluster

  std::vector<uint64_t> uniqueIds;
  //consider all hits from all clusters in the merged cluster
  for (const auto& clu : clusters_) {
    auto detIds_and_rows = clu->detIds_and_rows();

    for (const auto& hit : detIds_and_rows) {
      uint32_t rawId = hit.first;
      uint8_t row = hit.second.first;
      uint8_t col = hit.second.second;

      uint8_t row8 = static_cast<uint8_t>(std::clamp(static_cast<int>(row), 0, 15));
      uint8_t col8 = static_cast<uint8_t>(std::clamp(static_cast<int>(col), 0, 15));

      uint8_t rowcol = static_cast<uint8_t>((row8 << krcOffset) | col8);

      uint64_t uniqueId = (static_cast<uint64_t>(rawId) << 8) | static_cast<uint64_t>(rowcol);

      //avoid duplicates in the uniqueIds vector
      if (std::find(uniqueIds.begin(), uniqueIds.end(), uniqueId) == uniqueIds.end()) {
        uniqueIds.push_back(uniqueId);
      }
    }
  }

  return uniqueIds;
}

unsigned int MtdSimMergedCluster::hitProdType() const {
  unsigned int thisType(0), oldType(999);
  for (auto const& clu : clusters_) {
    thisType = std::min((*clu).hitProdType(), oldType);
    oldType = thisType;
  }
  return thisType;
}

std::ostream& operator<<(std::ostream& s, const MtdSimMergedCluster& sc) {
  s << "MtdSimMergedCluster with " << sc.clusters_.size()
    << " clusters and TrackingParticles: = " << sc.trackingParticles_.size() << "\n";
  s << "Earliest time = " << sc.simTime() << "\n";
  s << "Earliest position = " << sc.simPos() << "\n";

  return s;
}
