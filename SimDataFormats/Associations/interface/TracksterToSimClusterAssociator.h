#ifndef SimDataFormats_Associations_TracksterToSimClusterAssociator_h
#define SimDataFormats_Associations_TracksterToSimClusterAssociator_h
// Original Author:  Leonardo Cristella

// system include files
#include <memory>

// user include files

#include "SimDataFormats/Associations/interface/TracksterToSimClusterAssociatorBaseImpl.h"

// forward declarations

namespace hgcal {

  class TracksterToSimClusterAssociator {
  public:
    TracksterToSimClusterAssociator(std::unique_ptr<hgcal::TracksterToSimClusterAssociatorBaseImpl>);
    TracksterToSimClusterAssociator() = default;
    TracksterToSimClusterAssociator(TracksterToSimClusterAssociator &&) = default;
    TracksterToSimClusterAssociator &operator=(TracksterToSimClusterAssociator &&) = default;
    TracksterToSimClusterAssociator(const TracksterToSimClusterAssociator &) = delete;                   // stop default
    const TracksterToSimClusterAssociator &operator=(const TracksterToSimClusterAssociator &) = delete;  // stop default

    ~TracksterToSimClusterAssociator() = default;

    // ---------- const member functions ---------------------
    /// Associate a Trackster to SimClusters
    hgcal::RecoToSimCollectionTracksters associateRecoToSim(const edm::Handle<ticl::TracksterCollection> &tCH,
                                                            const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                                            const edm::Handle<SimClusterCollection> &sCCH) const {
      return m_impl->associateRecoToSim(tCH, lCCH, sCCH);
    };

    /// Associate a SimCluster to Tracksters
    hgcal::SimToRecoCollectionTracksters associateSimToReco(const edm::Handle<ticl::TracksterCollection> &tCH,
                                                            const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                                            const edm::Handle<SimClusterCollection> &sCCH) const {
      return m_impl->associateSimToReco(tCH, lCCH, sCCH);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<TracksterToSimClusterAssociatorBaseImpl> m_impl;
  };
}  // namespace hgcal

#endif
