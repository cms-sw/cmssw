#ifndef SimDataFormats_Associations_LayerClusterToSimClusterAssociator_h
#define SimDataFormats_Associations_LayerClusterToSimClusterAssociator_h
// Original Author:  Marco Rovere

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociatorBaseImpl.h"

// forward declarations

namespace ticl {

  class LayerClusterToSimClusterAssociator {
  public:
    LayerClusterToSimClusterAssociator(std::unique_ptr<ticl::LayerClusterToSimClusterAssociatorBaseImpl>);
    LayerClusterToSimClusterAssociator() = default;
    LayerClusterToSimClusterAssociator(LayerClusterToSimClusterAssociator &&) = default;
    LayerClusterToSimClusterAssociator &operator=(LayerClusterToSimClusterAssociator &&) = default;
    LayerClusterToSimClusterAssociator(const LayerClusterToSimClusterAssociator &) = delete;  // stop default

    ~LayerClusterToSimClusterAssociator() = default;
    const LayerClusterToSimClusterAssociator &operator=(const LayerClusterToSimClusterAssociator &) =
        delete;  // stop default
    // ---------- const member functions ---------------------
    /// Associate a LayerCluster to SimClusters
    ticl::RecoToSimCollectionWithSimClusters associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCCH,
                                                                const edm::Handle<SimClusterCollection> &sCCH) const {
      return m_impl->associateRecoToSim(cCCH, sCCH);
    };

    /// Associate a SimCluster to LayerClusters
    ticl::SimToRecoCollectionWithSimClusters associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCCH,
                                                                const edm::Handle<SimClusterCollection> &sCCH) const {
      return m_impl->associateSimToReco(cCCH, sCCH);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<LayerClusterToSimClusterAssociatorBaseImpl> m_impl;
  };
}  // namespace ticl

#endif
