#ifndef SimDataFormats_CaloAnalysis_MtdSimMergedCluster_h
#define SimDataFormats_CaloAnalysis_MtdSimMergedCluster_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include <vector>

class MtdSimMergedCluster {
  friend std::ostream& operator<<(std::ostream& s, const MtdSimMergedCluster& sc);

public:
  MtdSimMergedCluster() = default;

  // Construct with one TrackingParticle ref (the main track)
  MtdSimMergedCluster(const TrackingParticleRef& tpRef) {
    if (tpRef.isNonnull()) {
      mainTrack_ = tpRef;
      trackingParticles_.push_back(tpRef);
    }
  }

  // Construct with one MtdSimLayerCluster ref and one TrackingParticle ref
  MtdSimMergedCluster(const MtdSimLayerClusterRef& clusterRef, const TrackingParticleRef& tpRef);

  ~MtdSimMergedCluster() = default;

  void addCluster(const MtdSimLayerClusterRef& clusterRef, const TrackingParticleRef& tpRef);

  /// energy-weighted average of composing cluster times
  float simTime() const;

  /// Position of the earliest cluster
  LocalPoint simPos() const { return simMC_pos_; }

  /// Set position of the merged cluster
  void setSimPos(const LocalPoint& pos) { simMC_pos_ = pos; }

  /// Energy of mergedcluster
  float simEnergy() const;

  /// Retrieve the stored geographic module DetId
  DetId simDetId() const { return simMC_detId_; }

  /// Set the geographic module DetId
  void setSimDetId(const DetId& id) { simMC_detId_ = id; }

  /// Retrieve list of all DetIds from clusters
  std::vector<DetId> detIds() const;

  /// Retrieve list of times and positions of all sim hits in the clusters
  std::vector<std::pair<float, LocalPoint>> hitTimesAndPositions() const;

  /// Retrieve cluster production type
  /// if primary is present, use that
  unsigned int hitProdType() const;

  unsigned int seedHitProdType() const;

  /// Accessors
  const MtdSimLayerClusterRefVector& clusters() const { return clusters_; }
  const TrackingParticleRefVector& trackingParticles() const { return trackingParticles_; }

private:
  static constexpr uint32_t krcOffset = 4;
  MtdSimLayerClusterRefVector clusters_;
  TrackingParticleRefVector trackingParticles_;
  TrackingParticleRef mainTrack_;
  LocalPoint simMC_pos_ = LocalPoint(0.f, 0.f, 0.f);
  DetId simMC_detId_ = DetId(0);
};

#endif
