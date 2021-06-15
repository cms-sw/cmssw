#ifndef SimDataFormats_Associations_LayerClusterToSimTracksterAssociator_h
#define SimDataFormats_Associations_LayerClusterToSimTracksterAssociator_h
// Original Author:  Leonardo Cristella

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociatorBaseImpl.h"

// forward declarations

namespace hgcal {

  class LayerClusterToSimTracksterAssociator {
  public:
    LayerClusterToSimTracksterAssociator(std::unique_ptr<hgcal::LayerClusterToSimTracksterAssociatorBaseImpl>);
    LayerClusterToSimTracksterAssociator() = default;
    LayerClusterToSimTracksterAssociator(LayerClusterToSimTracksterAssociator &&) = default;
    LayerClusterToSimTracksterAssociator &operator=(LayerClusterToSimTracksterAssociator &&) = default;
    ~LayerClusterToSimTracksterAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate a LayerCluster to SimTracksters
    hgcal::RecoToSimTracksterCollection associateRecoToSim(
        const edm::Handle<reco::CaloClusterCollection> &cCCH,
        const edm::Handle<ticl::TracksterCollection> &stCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const hgcal::RecoToSimCollection &lCToCPs,
        const edm::Handle<SimClusterCollection> &sCCH,
        const hgcal::RecoToSimCollectionWithSimClusters &lCToSCs) const {
      return m_impl->associateRecoToSim(cCCH, stCH, cPCH, lCToCPs, sCCH, lCToSCs);
    };

    /// Associate a SimTrackster to LayerClusters
    hgcal::SimTracksterToRecoCollection associateSimToReco(
        const edm::Handle<reco::CaloClusterCollection> &cCCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const hgcal::SimToRecoCollection &cpToLCs,
        const edm::Handle<SimClusterCollection> &sCCH,
        const hgcal::SimToRecoCollectionWithSimClusters &sCToLCs) const {
      return m_impl->associateSimToReco(cCCH, sTCH, cPCH, cpToLCs, sCCH, sCToLCs);
    }

  private:
    LayerClusterToSimTracksterAssociator(const LayerClusterToSimTracksterAssociator &) = delete;  // stop default

    const LayerClusterToSimTracksterAssociator &operator=(const LayerClusterToSimTracksterAssociator &) =
        delete;  // stop default

    // ---------- member data --------------------------------
    std::unique_ptr<LayerClusterToSimTracksterAssociatorBaseImpl> m_impl;
  };
}  // namespace hgcal

#endif
