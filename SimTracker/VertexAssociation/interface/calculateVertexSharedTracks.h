#ifndef SimTracker_VertexAssociation_calculateVertexSharedTracks_h
#define SimTracker_VertexAssociation_calculateVertexSharedTracks_h

#include <vector>

#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

struct SharedTracksAndFractions {
  SharedTracksAndFractions(unsigned int nSharedTracks,
                           float sharedTracksFraction,
                           float sharedPt2Fraction,
                           float sharedDzErrFraction)
      : nSharedTracks_(nSharedTracks),
        sharedTracksFraction_(sharedTracksFraction),
        sharedPt2Fraction_(sharedPt2Fraction),
        sharedDzErrFraction_(sharedDzErrFraction) {}

  const unsigned int nSharedTracks_;
  const float sharedTracksFraction_;
  const float sharedPt2Fraction_;
  const float sharedDzErrFraction_;
};

// =============================================================================
// Multi-collection association map types
//
// Vertices — in particular those built from PF candidates — may contain tracks
// from different underlying collections (e.g. generalTracks for charged
// hadrons, gsfTracks for electrons). A single RecoToSimCollection or
// SimToRecoCollection covers exactly one source collection; lookups for tracks
// from any other collection silently return end() and are missed.
//
// The overloads below therefore accept vectors of association maps. Each track
// is looked up in every map in turn; the first map that returns a hit is used.
// Maps that do not cover a given track's collection are harmlessly skipped
// since AssociationMap::find() keyed by a ref from a different product returns
// end() without error.
//
// Single-collection overloads are retained for backward compatibility and are
// implemented as thin wrappers that forward to the vector overloads.
// =============================================================================

using RecoToSimCollectionVec = std::vector<const reco::RecoToSimCollection *>;
using SimToRecoCollectionVec = std::vector<const reco::SimToRecoCollection *>;

// -----------------------------------------------------------------------------
// reco::Vertex overloads
// -----------------------------------------------------------------------------

/// Multi-collection overloads — preferred for PF-based vertices.
SharedTracksAndFractions calculateVertexSharedTracks(const reco::Vertex &recoV,
                                                     const TrackingVertex &simV,
                                                     const RecoToSimCollectionVec &trackRecoToSimAssociations);

SharedTracksAndFractions calculateVertexSharedTracks(const TrackingVertex &simV,
                                                     const reco::Vertex &recoV,
                                                     const SimToRecoCollectionVec &trackSimToRecoAssociations);

/// Single-collection overloads — retained for backward compatibility.
/// Forwarded to the vector overloads internally.
inline SharedTracksAndFractions calculateVertexSharedTracks(
    const reco::Vertex &recoV, const TrackingVertex &simV, const reco::RecoToSimCollection &trackRecoToSimAssociation) {
  return calculateVertexSharedTracks(recoV, simV, RecoToSimCollectionVec{&trackRecoToSimAssociation});
}

inline SharedTracksAndFractions calculateVertexSharedTracks(
    const TrackingVertex &simV, const reco::Vertex &recoV, const reco::SimToRecoCollection &trackSimToRecoAssociation) {
  return calculateVertexSharedTracks(simV, recoV, SimToRecoCollectionVec{&trackSimToRecoAssociation});
}

// -----------------------------------------------------------------------------
// reco::VertexCompositePtrCandidate overloads
//
// Track extraction from candidate daughters is done via dynamic_cast
// (reco::PFCandidate on RECO/AOD, pat::PackedCandidate on MiniAOD).
// Daughters from which no track can be recovered (neutral particles) are
// excluded from both numerator and denominator — they carry no tracking
// information and should not penalise the shared-track fraction.
//
// The fraction denominators follow the same conventions as the reco::Vertex
// overloads: sharedTracksFraction_ is relative to the number of reco
// daughters with a recoverable track; pt2 and dzError fractions are weighted
// sums over those same daughters.
//
// Multiple association maps are particularly important for this type since
// PF-based vertices regularly mix track collections (e.g. charged hadrons
// from generalTracks, electrons from gsfTracks).
// -----------------------------------------------------------------------------

/// Multi-collection overloads — preferred.
SharedTracksAndFractions calculateVertexSharedTracks(const reco::VertexCompositePtrCandidate &recoV,
                                                     const TrackingVertex &simV,
                                                     const RecoToSimCollectionVec &trackRecoToSimAssociations);

SharedTracksAndFractions calculateVertexSharedTracks(const TrackingVertex &simV,
                                                     const reco::VertexCompositePtrCandidate &recoV,
                                                     const SimToRecoCollectionVec &trackSimToRecoAssociations);

/// Single-collection overloads — retained for backward compatibility.
inline SharedTracksAndFractions calculateVertexSharedTracks(const reco::VertexCompositePtrCandidate &recoV,
                                                            const TrackingVertex &simV,
                                                            const reco::RecoToSimCollection &trackRecoToSimAssociation) {
  return calculateVertexSharedTracks(recoV, simV, RecoToSimCollectionVec{&trackRecoToSimAssociation});
}

inline SharedTracksAndFractions calculateVertexSharedTracks(const TrackingVertex &simV,
                                                            const reco::VertexCompositePtrCandidate &recoV,
                                                            const reco::SimToRecoCollection &trackSimToRecoAssociation) {
  return calculateVertexSharedTracks(simV, recoV, SimToRecoCollectionVec{&trackSimToRecoAssociation});
}

#endif  // SimTracker_VertexAssociation_calculateVertexSharedTracks_h
