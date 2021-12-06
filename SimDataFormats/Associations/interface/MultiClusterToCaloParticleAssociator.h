#ifndef SimDataFormats_Associations_MultiClusterToCaloParticleAssociator_h
#define SimDataFormats_Associations_MultiClusterToCaloParticleAssociator_h
// Original Author:  Leonardo Cristella

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/MultiClusterToCaloParticleAssociatorBaseImpl.h"

// forward declarations

namespace hgcal {

  class MultiClusterToCaloParticleAssociator {
  public:
    MultiClusterToCaloParticleAssociator(std::unique_ptr<hgcal::MultiClusterToCaloParticleAssociatorBaseImpl>);
    MultiClusterToCaloParticleAssociator() = default;
    MultiClusterToCaloParticleAssociator(MultiClusterToCaloParticleAssociator &&) = default;
    MultiClusterToCaloParticleAssociator &operator=(MultiClusterToCaloParticleAssociator &&) = default;
    MultiClusterToCaloParticleAssociator(const MultiClusterToCaloParticleAssociator &) = delete;  // stop default
    const MultiClusterToCaloParticleAssociator &operator=(const MultiClusterToCaloParticleAssociator &) =
        delete;  // stop default

    ~MultiClusterToCaloParticleAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate a MultiCluster to CaloParticles
    hgcal::RecoToSimCollectionWithMultiClusters associateRecoToSim(
        const edm::Handle<reco::HGCalMultiClusterCollection> &cCCH,
        const edm::Handle<CaloParticleCollection> &cPCH) const {
      return m_impl->associateRecoToSim(cCCH, cPCH);
    };

    /// Associate a CaloParticle to MultiClusters
    hgcal::SimToRecoCollectionWithMultiClusters associateSimToReco(
        const edm::Handle<reco::HGCalMultiClusterCollection> &cCCH,
        const edm::Handle<CaloParticleCollection> &cPCH) const {
      return m_impl->associateSimToReco(cCCH, cPCH);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<MultiClusterToCaloParticleAssociatorBaseImpl> m_impl;
  };
}  // namespace hgcal

#endif
