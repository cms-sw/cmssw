#ifndef SimDataFormats_Associations_TrackToTrackingParticleAssociatorBaseImpl_h
#define SimDataFormats_Associations_TrackToTrackingParticleAssociatorBaseImpl_h

/** \class TrackToTrackingParticleAssociatorBaseImpl
 *  Base class for TrackToTrackingParticleAssociators. Methods take as input the
 * handle of Track and TrackingPArticle collections and return an AssociationMap
 * (oneToManyWithQuality)
 *
 *  \author magni, cerati
 */

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

namespace reco {
  class TrackToTrackingParticleAssociatorBaseImpl {
  public:
    /// Constructor
    TrackToTrackingParticleAssociatorBaseImpl();
    /// Destructor
    virtual ~TrackToTrackingParticleAssociatorBaseImpl();

    /// compare reco to sim the handle of reco::Track and TrackingParticle
    /// collections
    virtual reco::RecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<reco::Track>> &tCH,
                                                         const edm::Handle<TrackingParticleCollection> &tPCH) const;

    /// compare reco to sim the handle of reco::Track and TrackingParticle
    /// collections
    virtual reco::SimToRecoCollection associateSimToReco(const edm::Handle<edm::View<reco::Track>> &tCH,
                                                         const edm::Handle<TrackingParticleCollection> &tPCH) const;

    /// Association Reco To Sim with Collections
    virtual reco::RecoToSimCollection associateRecoToSim(
        const edm::RefToBaseVector<reco::Track> &tc, const edm::RefVector<TrackingParticleCollection> &tpc) const = 0;
    /// Association Sim To Reco with Collections
    virtual reco::SimToRecoCollection associateSimToReco(
        const edm::RefToBaseVector<reco::Track> &tc, const edm::RefVector<TrackingParticleCollection> &tpc) const = 0;

    // TrajectorySeed
    virtual reco::RecoToSimCollectionSeed associateRecoToSim(const edm::Handle<edm::View<TrajectorySeed>> &,
                                                             const edm::Handle<TrackingParticleCollection> &) const;

    virtual reco::SimToRecoCollectionSeed associateSimToReco(const edm::Handle<edm::View<TrajectorySeed>> &,
                                                             const edm::Handle<TrackingParticleCollection> &) const;

    // TrackCandidate
    virtual reco::RecoToSimCollectionTCandidate associateRecoToSim(
        const edm::Handle<TrackCandidateCollection> &, const edm::RefVector<TrackingParticleCollection> &) const;

    virtual reco::SimToRecoCollectionTCandidate associateSimToReco(
        const edm::Handle<TrackCandidateCollection> &, const edm::RefVector<TrackingParticleCollection> &) const;
  };
}  // namespace reco

#endif
