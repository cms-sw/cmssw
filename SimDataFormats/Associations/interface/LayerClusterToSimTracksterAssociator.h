#ifndef SimDataFormats_Associations_LayerClusterToSimTracksterAssociator_h
#define SimDataFormats_Associations_LayerClusterToSimTracksterAssociator_h
// Original Author:  Leonardo Cristella

// system include files
#include <memory>

// user include files
#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociatorBaseImpl.h"
#include "DataFormats/Common/interface/Uninitialized.h"

// forward declarations

namespace ticl {

  class LayerClusterToSimTracksterAssociator {
  public:
    LayerClusterToSimTracksterAssociator(std::unique_ptr<ticl::LayerClusterToSimTracksterAssociatorBaseImpl>);
    LayerClusterToSimTracksterAssociator() = delete;
    explicit LayerClusterToSimTracksterAssociator(edm::Uninitialized) noexcept {};
    LayerClusterToSimTracksterAssociator(LayerClusterToSimTracksterAssociator &&) = default;
    LayerClusterToSimTracksterAssociator &operator=(LayerClusterToSimTracksterAssociator &&) = default;
    LayerClusterToSimTracksterAssociator(const LayerClusterToSimTracksterAssociator &) = delete;  // stop default
    const LayerClusterToSimTracksterAssociator &operator=(const LayerClusterToSimTracksterAssociator &) =
        delete;  // stop default

    ~LayerClusterToSimTracksterAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate a LayerCluster to SimTracksters
    ticl::RecoToSimTracksterCollection associateRecoToSim(
        const edm::Handle<reco::CaloClusterCollection> &cCCH,
        const edm::Handle<ticl::TracksterCollection> &stCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const ticl::RecoToSimCollectionT<reco::CaloClusterCollection> &lCToCPs,
        const edm::Handle<SimClusterCollection> &sCCH,
        const ticl::RecoToSimCollectionWithSimClustersT<reco::CaloClusterCollection> &lCToSCs) const {
      return m_impl->associateRecoToSim(cCCH, stCH, cPCH, lCToCPs, sCCH, lCToSCs);
    };

    /// Associate a SimTrackster to LayerClusters
    ticl::SimTracksterToRecoCollection associateSimToReco(
        const edm::Handle<reco::CaloClusterCollection> &cCCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const ticl::SimToRecoCollectionT<reco::CaloClusterCollection> &cpToLCs,
        const edm::Handle<SimClusterCollection> &sCCH,
        const ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection> &sCToLCs) const {
      return m_impl->associateSimToReco(cCCH, sTCH, cPCH, cpToLCs, sCCH, sCToLCs);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<LayerClusterToSimTracksterAssociatorBaseImpl> m_impl;
  };
}  // namespace ticl

#endif
