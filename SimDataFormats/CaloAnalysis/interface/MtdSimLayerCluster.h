// Author: Aurora Perego, Fabio Cossutti - aurora.perego@cern.ch, fabio.cossutti@ts.infn.it
// Date: 05/2023

#ifndef SimDataFormats_CaloAnalysis_MtdSimLayerCluster_h
#define SimDataFormats_CaloAnalysis_MtdSimLayerCluster_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimCluster.h"
#include <vector>

class MtdSimLayerCluster : public MtdSimCluster {
  friend std::ostream &operator<<(std::ostream &s, MtdSimLayerCluster const &tp);

public:
  MtdSimLayerCluster();
  MtdSimLayerCluster(const SimTrack &simtrk);
  MtdSimLayerCluster(EncodedEventId eventID, uint32_t particleID);  // for PU

  // destructor
  ~MtdSimLayerCluster();

  /** @brief computes the time of the cluster */
  float computeClusterTime() {
    simLC_time_ = 0.;
    float tot_en = 0.;
    for (uint32_t i = 0; i < times_.size(); i++) {
      simLC_time_ += times_[i] * energies_[i];
      tot_en += energies_[i];
    }
    if (tot_en != 0.)
      simLC_time_ = simLC_time_ / tot_en;
    return simLC_time_;
  }

  /** @brief computes the energy of the cluster */
  void addCluEnergy(float energy) { simLC_energy_ = energy; }

  /** @brief computes the position of the cluster */
  void addCluLocalPos(LocalPoint pos) { simLC_pos_ = pos; }

  /** @brief add the index of the simcluster */
  void addCluIndex(const uint32_t index) { seedId_ = index; }

  /** @brief returns the time of the cluster */
  float simLCTime() const { return simLC_time_; }

  /** @brief returns the local position of the cluster */
  LocalPoint simLCPos() const { return simLC_pos_; }

  /** @brief returns the accumulated sim energy in the cluster */
  float simLCEnergy() const { return simLC_energy_; }

  uint32_t seedId() const { return seedId_; }

private:
  // id of the simCluster it comes from
  uint32_t seedId_;

  float simLC_time_{0.f};
  float simLC_energy_{0.f};
  LocalPoint simLC_pos_;
};

#endif
