#ifndef SimDataFormats_MtdCaloParticle_h
#define SimDataFormats_MtdCaloParticle_h

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimClusterFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <vector>

class SimTrack;
class EncodedEventId;

class MtdCaloParticle : public CaloParticle {
  friend std::ostream &operator<<(std::ostream &s, MtdCaloParticle const &tp);

public:
  typedef MtdSimClusterRefVector::iterator mtdsc_iterator;

  MtdCaloParticle();

  MtdCaloParticle(const SimTrack &simtrk);
  MtdCaloParticle(EncodedEventId eventID, uint32_t particleID);  // for PU

  // destructor
  ~MtdCaloParticle();

  void addSimCluster(const MtdSimClusterRef &ref) { mtdsimClusters_.push_back(ref); }

  /// iterators
  mtdsc_iterator simCluster_begin() const { return mtdsimClusters_.begin(); }
  mtdsc_iterator simCluster_end() const { return mtdsimClusters_.end(); }

  const MtdSimClusterRefVector &simClusters() const { return mtdsimClusters_; }
  void clearSimClusters() { mtdsimClusters_.clear(); }

  /** @brief returns the time of the caloparticle */
  float simTime() const { return simhit_time_; }

  void addSimTime(const float time) { simhit_time_ = time; }

  /** @brief add simhit's energy to cluster */
  void addSimHit(PSimHit &hit) {
    simhit_energy_ += hit.energyLoss();
    ++nsimhits_;
  }

private:
  float simhit_time_{-99.f};
  MtdSimClusterRefVector mtdsimClusters_;
};

#endif  // SimDataFormats_MtdCaloParticle_H
