#ifndef SimDataFormats_Associations_LayerClusterToSimClusterAssociator_h
#define SimDataFormats_Associations_LayerClusterToSimClusterAssociator_h
// Original Author:  Marco Rovere

// system include files
#include <memory>

// user include files
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociatorBaseImpl.h"
#include "DataFormats/Common/interface/Uninitialized.h"

// forward declarations

namespace ticl {

  template <typename CLUSTER>
  class LayerClusterToSimClusterAssociatorT {
  public:
    explicit LayerClusterToSimClusterAssociatorT(
        std::unique_ptr<LayerClusterToSimClusterAssociatorBaseImplT<CLUSTER>> impl)
        : m_impl(std::move(impl)) {}
    LayerClusterToSimClusterAssociatorT() = delete;
    explicit LayerClusterToSimClusterAssociatorT(edm::Uninitialized) noexcept {};
    LayerClusterToSimClusterAssociatorT(LayerClusterToSimClusterAssociatorT<CLUSTER> &&) = default;
    LayerClusterToSimClusterAssociatorT &operator=(LayerClusterToSimClusterAssociatorT<CLUSTER> &&) = default;
    LayerClusterToSimClusterAssociatorT(const LayerClusterToSimClusterAssociatorT<CLUSTER> &) = delete;  // stop default

    ~LayerClusterToSimClusterAssociatorT() = default;
    const LayerClusterToSimClusterAssociatorT &operator=(const LayerClusterToSimClusterAssociatorT<CLUSTER> &) =
        delete;  // stop default
    // ---------- const member functions ---------------------
    /// Associate a LayerCluster to SimClusters
    RecoToSimCollectionWithSimClustersT<CLUSTER> associateRecoToSim(
        const edm::Handle<CLUSTER> &cCCH, const edm::Handle<SimClusterCollection> &sCCH) const {
      return m_impl->associateRecoToSim(cCCH, sCCH);
    };

    /// Associate a SimCluster to LayerClusters
    SimToRecoCollectionWithSimClustersT<CLUSTER> associateSimToReco(
        const edm::Handle<CLUSTER> &cCCH, const edm::Handle<SimClusterCollection> &sCCH) const {
      return m_impl->associateSimToReco(cCCH, sCCH);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<LayerClusterToSimClusterAssociatorBaseImplT<CLUSTER>> m_impl;
  };
}  // namespace ticl

extern template class ticl::LayerClusterToSimClusterAssociatorT<reco::CaloClusterCollection>;
extern template class ticl::LayerClusterToSimClusterAssociatorT<reco::PFClusterCollection>;

using LayerClusterToSimClusterAssociator = ticl::LayerClusterToSimClusterAssociatorT<reco::CaloClusterCollection>;
using ParticleFlowClusterToSimClusterAssociator = ticl::LayerClusterToSimClusterAssociatorT<reco::PFClusterCollection>;

#endif
