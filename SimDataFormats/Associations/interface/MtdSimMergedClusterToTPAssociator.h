#ifndef SimDataFormats_Associations_MtdSimMergedClusterToTPAssociator_h
#define SimDataFormats_Associations_MtdSimMergedClusterToTPAssociator_h

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/MtdSimMergedClusterToTPAssociatorBaseImpl.h"

// forward declarations

namespace reco {
  class MtdSimMergedClusterToTPAssociator {
  public:
    MtdSimMergedClusterToTPAssociator(std::unique_ptr<reco::MtdSimMergedClusterToTPAssociatorBaseImpl>);
    MtdSimMergedClusterToTPAssociator() = default;
    MtdSimMergedClusterToTPAssociator(MtdSimMergedClusterToTPAssociator &&) = default;
    MtdSimMergedClusterToTPAssociator &operator=(MtdSimMergedClusterToTPAssociator &&) = default;
    MtdSimMergedClusterToTPAssociator(const MtdSimMergedClusterToTPAssociator &) = delete;  // stop default
    const MtdSimMergedClusterToTPAssociator &operator=(const MtdSimMergedClusterToTPAssociator &) =
        delete;  // stop default

    ~MtdSimMergedClusterToTPAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate MtdSimMergedCluster to TrackingParticle
    reco::MergedSimToTPCollectionMtd associateSimToTP(
        const edm::Handle<MtdSimMergedClusterCollection> &simClusH,
        const edm::Handle<TrackingParticleCollection> &trackingParticleH) const {
      return m_impl->associateSimToTP(simClusH, trackingParticleH);
    };

    /// Associate TrackingParticle to MtdSimMergedCluster
    reco::TPToMergedSimCollectionMtd associateTPToSim(
        const edm::Handle<MtdSimMergedClusterCollection> &simClusH,
        const edm::Handle<TrackingParticleCollection> &trackingParticleH) const {
      return m_impl->associateTPToSim(simClusH, trackingParticleH);
    };

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<MtdSimMergedClusterToTPAssociatorBaseImpl> m_impl;
  };
}  // namespace reco

#endif
