#ifndef SimDataFormats_Associations_TrackToTrackingParticleAssociator_h
#define SimDataFormats_Associations_TrackToTrackingParticleAssociator_h
// -*- C++ -*-
//
// Package:     SimDataFormats/Associations
// Class  :     TrackToTrackingParticleAssociator
//
/**\class TrackToTrackingParticleAssociator TrackToTrackingParticleAssociator.h
 "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

 Description: Interface for accessing a Track to TrackingParticle associator

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 30 Dec 2014 20:47:00 GMT
//

// system include files
#include <memory>

// user include files
#include "SimDataFormats/Associations/interface/TrackAssociation.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociatorBaseImpl.h"

// forward declarations

namespace reco {
  class TrackToTrackingParticleAssociator {
  public:
#ifndef __GCCXML__
    TrackToTrackingParticleAssociator(std::unique_ptr<reco::TrackToTrackingParticleAssociatorBaseImpl>);
#endif
    TrackToTrackingParticleAssociator() = default;
    TrackToTrackingParticleAssociator(TrackToTrackingParticleAssociator &&) = default;
    TrackToTrackingParticleAssociator &operator=(TrackToTrackingParticleAssociator &&) = default;
    TrackToTrackingParticleAssociator(const TrackToTrackingParticleAssociator &) = delete;  // stop default
    const TrackToTrackingParticleAssociator &operator=(const TrackToTrackingParticleAssociator &) =
        delete;  // stop default

    ~TrackToTrackingParticleAssociator() = default;

    // ---------- const member functions ---------------------
    /// compare reco to sim the handle of reco::Track and TrackingParticle
    /// collections
    reco::RecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<reco::Track>> &tCH,
                                                 const edm::Handle<TrackingParticleCollection> &tPCH) const {
      return m_impl->associateRecoToSim(tCH, tPCH);
    }

    /// compare reco to sim the handle of reco::Track and TrackingParticle
    /// collections
    reco::SimToRecoCollection associateSimToReco(const edm::Handle<edm::View<reco::Track>> &tCH,
                                                 const edm::Handle<TrackingParticleCollection> &tPCH) const {
      return m_impl->associateSimToReco(tCH, tPCH);
    }

    /// Association Reco To Sim with Collections
    reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track> &tc,
                                                 const edm::RefVector<TrackingParticleCollection> &tpc) const {
      return m_impl->associateRecoToSim(tc, tpc);
    }

    /// Association Sim To Reco with Collections
    reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track> &tc,
                                                 const edm::RefVector<TrackingParticleCollection> &tpc) const {
      return m_impl->associateSimToReco(tc, tpc);
    }

    // TrajectorySeed
    reco::RecoToSimCollectionSeed associateRecoToSim(const edm::Handle<edm::View<TrajectorySeed>> &ts,
                                                     const edm::Handle<TrackingParticleCollection> &tpc) const {
      return m_impl->associateRecoToSim(ts, tpc);
    }

    reco::SimToRecoCollectionSeed associateSimToReco(const edm::Handle<edm::View<TrajectorySeed>> &ts,
                                                     const edm::Handle<TrackingParticleCollection> &tpc) const {
      return m_impl->associateSimToReco(ts, tpc);
    }

    // TrackCandidate
    reco::RecoToSimCollectionTCandidate associateRecoToSim(
        const edm::Handle<TrackCandidateCollection> &tc, const edm::RefVector<TrackingParticleCollection> &tpc) const {
      return m_impl->associateRecoToSim(tc, tpc);
    }

    reco::SimToRecoCollectionTCandidate associateSimToReco(
        const edm::Handle<TrackCandidateCollection> &tc, const edm::RefVector<TrackingParticleCollection> &tpc) const {
      return m_impl->associateSimToReco(tc, tpc);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<TrackToTrackingParticleAssociatorBaseImpl> m_impl;
  };
}  // namespace reco

#endif
