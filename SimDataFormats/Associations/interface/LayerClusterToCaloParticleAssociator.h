#ifndef SimDataFormats_Associations_LayerClusterToCaloParticleAssociator_h
#define SimDataFormats_Associations_LayerClusterToCaloParticleAssociator_h
// Original Author:  Marco Rovere

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociatorBaseImpl.h"

// forward declarations

namespace ticl {

  class LayerClusterToCaloParticleAssociator {
  public:
    LayerClusterToCaloParticleAssociator(std::unique_ptr<ticl::LayerClusterToCaloParticleAssociatorBaseImpl>);
    LayerClusterToCaloParticleAssociator() = default;
    LayerClusterToCaloParticleAssociator(LayerClusterToCaloParticleAssociator &&) = default;
    LayerClusterToCaloParticleAssociator &operator=(LayerClusterToCaloParticleAssociator &&) = default;
    LayerClusterToCaloParticleAssociator(const LayerClusterToCaloParticleAssociator &) = delete;  // stop default
    const LayerClusterToCaloParticleAssociator &operator=(const LayerClusterToCaloParticleAssociator &) =
        delete;  // stop default

    ~LayerClusterToCaloParticleAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate a LayerCluster to CaloParticles
    ticl::RecoToSimCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCCH,
                                                 const edm::Handle<CaloParticleCollection> &cPCH) const {
      return m_impl->associateRecoToSim(cCCH, cPCH);
    };

    /// Associate a CaloParticle to LayerClusters
    ticl::SimToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCCH,
                                                 const edm::Handle<CaloParticleCollection> &cPCH) const {
      return m_impl->associateSimToReco(cCCH, cPCH);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<LayerClusterToCaloParticleAssociatorBaseImpl> m_impl;
  };
}  // namespace ticl

#endif
