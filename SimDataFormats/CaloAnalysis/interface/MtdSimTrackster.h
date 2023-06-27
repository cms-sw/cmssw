// Author: Aurora Perego, Fabio Cossutti - aurora.perego@cern.ch, fabio.cossutti@ts.infn.it
// Date: 05/2023

#ifndef SimDataFormats_MtdMtdSimTrackster_h
#define SimDataFormats_MtdMtdSimTrackster_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimCluster.h"
#include <vector>

class MtdSimTrackster : public MtdSimCluster {
  friend std::ostream &operator<<(std::ostream &s, MtdSimTrackster const &tp);

public:
  MtdSimTrackster();

  MtdSimTrackster(const SimCluster &sc);
  MtdSimTrackster(EncodedEventId eventID, uint32_t particleID);  // for PU
  MtdSimTrackster(const SimCluster &sc, const std::vector<uint32_t> SCs, const float time, const GlobalPoint pos);

  // destructor
  ~MtdSimTrackster();

  /** @brief returns the position of the cluster */
  GlobalPoint position() const { return posAtEntrance_; }

  /** @brief returns the time of the cluster */
  float time() const { return timeAtEntrance_; }

  /** @brief returns the layer clusters indexes in the sim trackster*/
  std::vector<uint32_t> clusters() const { return clusters_; }

  /** @brief add simhit's energy to cluster */
  void addCluster(const uint32_t sc) { clusters_.push_back(sc); }

  /** @brief Gives the total number of SimHits, in the cluster */
  int numberOfClusters() const { return clusters_.size(); }

private:
  float timeAtEntrance_{0.f};
  GlobalPoint posAtEntrance_;
  // indices of the MtdSimLayerClusters contained in the simTrackster
  std::vector<uint32_t> clusters_;
};

#endif  // SimDataFormats_MtdSimTrackster_H
