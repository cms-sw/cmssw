#ifndef SimDataFormats_Associations_LayerClusterToCaloParticleAssociator_h
#define SimDataFormats_Associations_LayerClusterToCaloParticleAssociator_h
// Original Author:  Marco Rovere

// system include files
#include <memory>

// user include files
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociatorBaseImpl.h"
#include "DataFormats/Common/interface/Uninitialized.h"

// forward declarations

namespace ticl {

  template <typename CLUSTER>
  class LayerClusterToCaloParticleAssociatorT {
  public:
    explicit LayerClusterToCaloParticleAssociatorT(
        std::unique_ptr<ticl::LayerClusterToCaloParticleAssociatorBaseImplT<CLUSTER>> impl)
        : m_impl(std::move(impl)) {}
    LayerClusterToCaloParticleAssociatorT() = delete;
    explicit LayerClusterToCaloParticleAssociatorT(edm::Uninitialized) noexcept {};
    LayerClusterToCaloParticleAssociatorT(LayerClusterToCaloParticleAssociatorT<CLUSTER> &&) = default;
    LayerClusterToCaloParticleAssociatorT &operator=(LayerClusterToCaloParticleAssociatorT<CLUSTER> &&) = default;
    LayerClusterToCaloParticleAssociatorT(const LayerClusterToCaloParticleAssociatorT<CLUSTER> &) =
        delete;  // stop default
    const LayerClusterToCaloParticleAssociatorT &operator=(const LayerClusterToCaloParticleAssociatorT<CLUSTER> &) =
        delete;  // stop default

    ~LayerClusterToCaloParticleAssociatorT() = default;

    // ---------- const member functions ---------------------
    /// Associate a LayerCluster to CaloParticles
    ticl::RecoToSimCollectionT<CLUSTER> associateRecoToSim(const edm::Handle<CLUSTER> &cCCH,
                                                           const edm::Handle<CaloParticleCollection> &cPCH) const {
      return m_impl->associateRecoToSim(cCCH, cPCH);
    };

    /// Associate a CaloParticle to LayerClusters
    ticl::SimToRecoCollectionT<CLUSTER> associateSimToReco(const edm::Handle<CLUSTER> &cCCH,
                                                           const edm::Handle<CaloParticleCollection> &cPCH) const {
      return m_impl->associateSimToReco(cCCH, cPCH);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<LayerClusterToCaloParticleAssociatorBaseImplT<CLUSTER>> m_impl;
  };
}  // namespace ticl

extern template class ticl::LayerClusterToCaloParticleAssociatorT<reco::CaloClusterCollection>;
extern template class ticl::LayerClusterToCaloParticleAssociatorT<reco::PFClusterCollection>;

#endif
