#ifndef SimDataFormats_Associations_LayerClusterToSimTracksterAssociator_h
#define SimDataFormats_Associations_LayerClusterToSimTracksterAssociator_h
// Original Author:  Marco Rovere

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
    hgcal::RecoToSimCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCCH,
                                                  const edm::Handle<ticl::TracksterCollection> &stCH) const {
      return m_impl->associateRecoToSim(cCCH, stCH);
    };

    /// Associate a SimTrackster to LayerClusters
    hgcal::SimToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCCH,
                                                  const edm::Handle<CaloParticleCollection> &cPCH) const {
      return m_impl->associateSimToReco(cCCH, cPCH);
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
