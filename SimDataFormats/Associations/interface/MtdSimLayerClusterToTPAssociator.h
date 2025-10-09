#ifndef SimDataFormats_Associations_MtdSimLayerClusterToTPAssociator_h
#define SimDataFormats_Associations_MtdSimLayerClusterToTPAssociator_h
// Author:  M. Malberti

// system include files
#include <memory>

// user include files
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"
#include "DataFormats/Common/interface/Uninitialized.h"

// forward declarations

namespace reco {
  class MtdSimLayerClusterToTPAssociator {
  public:
    MtdSimLayerClusterToTPAssociator(std::unique_ptr<reco::MtdSimLayerClusterToTPAssociatorBaseImpl>);
    MtdSimLayerClusterToTPAssociator() = delete;
    explicit MtdSimLayerClusterToTPAssociator(edm::Uninitialized) noexcept {};
    MtdSimLayerClusterToTPAssociator(MtdSimLayerClusterToTPAssociator &&) = default;
    MtdSimLayerClusterToTPAssociator &operator=(MtdSimLayerClusterToTPAssociator &&) = default;
    MtdSimLayerClusterToTPAssociator(const MtdSimLayerClusterToTPAssociator &) = delete;  // stop default
    const MtdSimLayerClusterToTPAssociator &operator=(const MtdSimLayerClusterToTPAssociator &) =
        delete;  // stop default

    ~MtdSimLayerClusterToTPAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate MtdSimLayerCluster to TrackingParticle
    reco::SimToTPCollectionMtd associateSimToTP(const edm::Handle<MtdSimLayerClusterCollection> &simClusH,
                                                const edm::Handle<TrackingParticleCollection> &trackingParticleH) const {
      return m_impl->associateSimToTP(simClusH, trackingParticleH);
    };

    /// Associate TrackingParticle to MtdSimLayerCluster
    reco::TPToSimCollectionMtd associateTPToSim(const edm::Handle<MtdSimLayerClusterCollection> &simClusH,
                                                const edm::Handle<TrackingParticleCollection> &trackingParticleH) const {
      return m_impl->associateTPToSim(simClusH, trackingParticleH);
    };

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<MtdSimLayerClusterToTPAssociatorBaseImpl> m_impl;
  };
}  // namespace reco

#endif
