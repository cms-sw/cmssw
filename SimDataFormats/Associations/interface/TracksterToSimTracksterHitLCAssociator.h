#ifndef SimDataFormats_Associations_TracksterToSimTracksterHitLCAssociator_h
#define SimDataFormats_Associations_TracksterToSimTracksterHitLCAssociator_h
// Original Author:  Leonardo Cristella

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociatorBaseImpl.h"

namespace hgcal {

  class TracksterToSimTracksterHitLCAssociator {
  public:
    TracksterToSimTracksterHitLCAssociator(std::unique_ptr<hgcal::TracksterToSimTracksterHitLCAssociatorBaseImpl>);
    TracksterToSimTracksterHitLCAssociator() = default;
    TracksterToSimTracksterHitLCAssociator(TracksterToSimTracksterHitLCAssociator &&) = default;
    TracksterToSimTracksterHitLCAssociator &operator=(TracksterToSimTracksterHitLCAssociator &&) = default;
    TracksterToSimTracksterHitLCAssociator(const TracksterToSimTracksterHitLCAssociator &) = delete;
    const TracksterToSimTracksterHitLCAssociator &operator=(const TracksterToSimTracksterHitLCAssociator &) = delete;

    ~TracksterToSimTracksterHitLCAssociator() = default;

    hgcal::association_t makeConnections(const edm::Handle<ticl::TracksterCollection> &tCH,
                                         const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                         const edm::Handle<SimClusterCollection> &sCCH,
                                         const edm::Handle<CaloParticleCollection> &cPCH,
                                         const edm::Handle<ticl::TracksterCollection> &sTCH) const {
      return m_impl->makeConnections(tCH, lCCH, sCCH, cPCH, sTCH);
    }
    /// Associate a Trackster to SimClusters
    hgcal::RecoToSimCollectionSimTracksters associateRecoToSim(
        const edm::Handle<ticl::TracksterCollection> &tCH,
        const edm::Handle<reco::CaloClusterCollection> &lCCH,
        const edm::Handle<SimClusterCollection> &sCCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH) const {
      return m_impl->associateRecoToSim(tCH, lCCH, sCCH, cPCH, sTCH);
    };

    /// Associate a SimCluster to Tracksters
    hgcal::SimToRecoCollectionSimTracksters associateSimToReco(
        const edm::Handle<ticl::TracksterCollection> &tCH,
        const edm::Handle<reco::CaloClusterCollection> &lCCH,
        const edm::Handle<SimClusterCollection> &sCCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH) const {
      return m_impl->associateSimToReco(tCH, lCCH, sCCH, cPCH, sTCH);
    }

  private:
    std::unique_ptr<TracksterToSimTracksterHitLCAssociatorBaseImpl> m_impl;
  };
}  // namespace hgcal

#endif
