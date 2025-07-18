#ifndef SimDataFormats_Associations_TracksterToSimTracksterAssociator_h
#define SimDataFormats_Associations_TracksterToSimTracksterAssociator_h
// Original Author:  Leonardo Cristella

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterAssociatorBaseImpl.h"

// forward declarations

namespace ticl {

  class TracksterToSimTracksterAssociator {
  public:
    TracksterToSimTracksterAssociator(std::unique_ptr<ticl::TracksterToSimTracksterAssociatorBaseImpl>);
    TracksterToSimTracksterAssociator() = default;
    TracksterToSimTracksterAssociator(TracksterToSimTracksterAssociator &&) = default;
    TracksterToSimTracksterAssociator &operator=(TracksterToSimTracksterAssociator &&) = default;
    TracksterToSimTracksterAssociator(const TracksterToSimTracksterAssociator &) = delete;  // stop default
    const TracksterToSimTracksterAssociator &operator=(const TracksterToSimTracksterAssociator &) =
        delete;  // stop default

    ~TracksterToSimTracksterAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate a Trackster to SimClusters
    ticl::RecoToSimCollectionSimTracksters associateRecoToSim(const edm::Handle<ticl::TracksterCollection> &tCH,
                                                              const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                                              const edm::Handle<ticl::TracksterCollection> &sTCH) const {
      return m_impl->associateRecoToSim(tCH, lCCH, sTCH);
    };

    /// Associate a SimCluster to Tracksters
    ticl::SimToRecoCollectionSimTracksters associateSimToReco(const edm::Handle<ticl::TracksterCollection> &tCH,
                                                              const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                                              const edm::Handle<ticl::TracksterCollection> &sTCH) const {
      return m_impl->associateSimToReco(tCH, lCCH, sTCH);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<TracksterToSimTracksterAssociatorBaseImpl> m_impl;
  };
}  // namespace ticl

#endif
