#ifndef SimDataFormats_Associations_LayerClusterToSimClusterAssociator_h
#define SimDataFormats_Associations_LayerClusterToSimClusterAssociator_h
// Original Author:  Marco Rovere

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociatorBaseImpl.h"

// forward declarations

namespace hgcal {

  class LayerClusterToSimClusterAssociator {
  public:
    LayerClusterToSimClusterAssociator(std::unique_ptr<hgcal::LayerClusterToSimClusterAssociatorBaseImpl>);
    LayerClusterToSimClusterAssociator() = default;
    LayerClusterToSimClusterAssociator(LayerClusterToSimClusterAssociator &&) = default;
    LayerClusterToSimClusterAssociator &operator=(LayerClusterToSimClusterAssociator &&) = default;
    ~LayerClusterToSimClusterAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate a LayerCluster to SimClusters
    hgcal::RecoToSimCollectionWithSimClusters associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCCH,
                                                                 const edm::Handle<SimClusterCollection> &sCCH) const {
      return m_impl->associateRecoToSim(cCCH, sCCH);
    };

    /// Associate a SimCluster to LayerClusters
    hgcal::SimToRecoCollectionWithSimClusters associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCCH,
                                                                 const edm::Handle<SimClusterCollection> &sCCH) const {
      return m_impl->associateSimToReco(cCCH, sCCH);
    }

  private:
    LayerClusterToSimClusterAssociator(const LayerClusterToSimClusterAssociator &) = delete;  // stop default

    const LayerClusterToSimClusterAssociator &operator=(const LayerClusterToSimClusterAssociator &) =
        delete;  // stop default

    // ---------- member data --------------------------------
    std::unique_ptr<LayerClusterToSimClusterAssociatorBaseImpl> m_impl;
  };
}  // namespace hgcal

#endif
