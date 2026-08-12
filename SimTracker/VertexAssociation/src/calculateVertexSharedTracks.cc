#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// =============================================================================
// Internal helpers (file-local)
// =============================================================================

namespace {

  /// Extract a TrackRef from a reco::Candidate daughter.
  /// Tries reco::PFCandidate (RECO/AOD) then pat::PackedCandidate (MiniAOD).
  /// Returns an invalid TrackRef for neutral or unresolvable daughters.
  ///
  /// Note on MiniAOD: pat::PackedCandidate::bestTrack() returns a raw pointer
  /// to an embedded track that cannot be wrapped into a persistent TrackRef
  /// usable by the standard track associators. Such daughters are therefore
  /// treated as unresolvable and excluded from the shared-track calculation.
  /// TODO: investigate transient TrackRef construction from packed track
  /// collection to enable full track-content association on MiniAOD.
  reco::TrackRef extractTrackRef(const reco::Candidate &cand) {
    if (const auto *pfc = dynamic_cast<const reco::PFCandidate *>(&cand))
      return pfc->trackRef();

    if (const auto *pkd = dynamic_cast<const pat::PackedCandidate *>(&cand)) {
      (void)pkd;  // suppress unused-variable warning pending the TODO above
      return reco::TrackRef();
    }

    return reco::TrackRef();
  }

  /// Collect all recoverable TrackRefs from a VertexCompositePtrCandidate.
  /// The returned vector contains only non-null refs; neutral daughters are
  /// silently skipped and do not appear in the output.
  std::vector<reco::TrackRef> extractTrackRefs(const reco::VertexCompositePtrCandidate &vtx) {
    std::vector<reco::TrackRef> refs;
    refs.reserve(vtx.numberOfDaughters());
    for (size_t i = 0; i < vtx.numberOfDaughters(); ++i) {
      const reco::Candidate *dau = vtx.daughter(i);
      if (!dau)
        continue;
      reco::TrackRef ref = extractTrackRef(*dau);
      if (ref.isNonnull())
        refs.push_back(ref);
    }
    return refs;
  }

  /// Look up a TrackBaseRef across a vector of RecoToSimCollections.
  /// Returns a pointer to the matched value range in the first map that
  /// contains the key, or nullptr if no map covers this track.
  /// Maps covering a different track collection return end() harmlessly.
  const reco::RecoToSimCollection::result_type *findInRecoToSimCollections(const reco::TrackBaseRef &trkRef,
                                                                           const RecoToSimCollectionVec &maps) {
    for (const reco::RecoToSimCollection *map : maps) {
      auto found = map->find(trkRef);
      if (found != map->end())
        return &found->val;
    }
    return nullptr;
  }

  /// Look up a TrackingParticleRef across a vector of SimToRecoCollections.
  /// Returns a pointer to the matched value range in the first map that
  /// contains the key, or nullptr if no map covers this TrackingParticle.
  const reco::SimToRecoCollection::result_type *findInSimToRecoCollections(const TrackingParticleRef &tpRef,
                                                                           const SimToRecoCollectionVec &maps) {
    for (const reco::SimToRecoCollection *map : maps) {
      auto found = map->find(tpRef);
      if (found != map->end())
        return &found->val;
    }
    return nullptr;
  }

}  // namespace

// =============================================================================
// reco::Vertex overloads — multi-collection
// =============================================================================

SharedTracksAndFractions calculateVertexSharedTracks(const reco::Vertex &recoV,
                                                     const TrackingVertex &simV,
                                                     const RecoToSimCollectionVec &trackRecoToSimAssociations) {
  unsigned int nSharedTracks = 0;
  float sharedTracksWeightPtSum2 = 0;
  float totalTracksWeightPtSum2 = 0;
  float sharedTracksWeightDzError = 0;
  float totalTracksWeightDzError = 0;

  for (auto iTrack = recoV.tracks_begin(); iTrack != recoV.tracks_end(); ++iTrack) {
    totalTracksWeightDzError += 1.0 / ((*iTrack)->dzError() * (*iTrack)->dzError());
    totalTracksWeightPtSum2 += (*iTrack)->pt() * (*iTrack)->pt();

    const auto *tpCandidates = findInRecoToSimCollections(reco::TrackBaseRef(*iTrack), trackRecoToSimAssociations);
    if (!tpCandidates)
      continue;

    // matched TP equal to any TP of sim vertex => increase counter
    for (const auto &tp : *tpCandidates) {
      if (std::find_if(simV.daughterTracks_begin(), simV.daughterTracks_end(), [&](const TrackingParticleRef &vtp) {
            return tp.first == vtp;
          }) != simV.daughterTracks_end()) {
        nSharedTracks += 1;
        sharedTracksWeightDzError += 1.0 / ((*iTrack)->dzError() * (*iTrack)->dzError());
        sharedTracksWeightPtSum2 += (*iTrack)->pt() * (*iTrack)->pt();
        break;
      }
    }
  }

  float sharedTracksFraction = (recoV.tracksSize() > 0) ? (float(nSharedTracks) / recoV.tracksSize()) : 0.0f;
  float sharedPt2Fraction = (totalTracksWeightPtSum2 > 0) ? (sharedTracksWeightPtSum2 / totalTracksWeightPtSum2) : 0.0f;
  float sharedDzErrFraction =
      (totalTracksWeightDzError > 0) ? (sharedTracksWeightDzError / totalTracksWeightDzError) : 0.0f;

  return SharedTracksAndFractions(nSharedTracks, sharedTracksFraction, sharedPt2Fraction, sharedDzErrFraction);
}

SharedTracksAndFractions calculateVertexSharedTracks(const TrackingVertex &simV,
                                                     const reco::Vertex &recoV,
                                                     const SimToRecoCollectionVec &trackSimToRecoAssociations) {
  unsigned int nSharedTracks = 0;
  float sharedTracksWeightPtSum2 = 0;
  float totalTracksWeightPtSum2 = 0;
  float sharedTracksWeightDzError = 0;
  float totalTracksWeightDzError = 0;

  // Total weights computed from reco side — denominator is always reco tracks.
  for (auto iTrack = recoV.tracks_begin(); iTrack != recoV.tracks_end(); ++iTrack) {
    totalTracksWeightPtSum2 += (*iTrack)->pt() * (*iTrack)->pt();
    totalTracksWeightDzError += 1.0 / ((*iTrack)->dzError() * (*iTrack)->dzError());
  }

  for (auto iTP = simV.daughterTracks_begin(); iTP != simV.daughterTracks_end(); ++iTP) {
    const auto *tkCandidates = findInSimToRecoCollections(*iTP, trackSimToRecoAssociations);
    if (!tkCandidates)
      continue;

    // matched track equal to any track of reco vertex => increase counter
    for (const auto &tk : *tkCandidates) {
      if (std::find_if(recoV.tracks_begin(), recoV.tracks_end(), [&](const reco::TrackBaseRef &vtk) {
            return (tk.first.id() == vtk.id()) && (tk.first.key() == vtk.key());
          }) != recoV.tracks_end()) {
        nSharedTracks += 1;
        sharedTracksWeightDzError += 1.0 / (tk.first->dzError() * tk.first->dzError());
        sharedTracksWeightPtSum2 += tk.first->pt() * tk.first->pt();
        break;
      }
    }
  }

  float sharedTracksFraction = (recoV.tracksSize() > 0) ? (float(nSharedTracks) / recoV.tracksSize()) : 0.0f;
  float sharedPt2Fraction = (totalTracksWeightPtSum2 > 0) ? (sharedTracksWeightPtSum2 / totalTracksWeightPtSum2) : 0.0f;
  float sharedDzErrFraction =
      (totalTracksWeightDzError > 0) ? (sharedTracksWeightDzError / totalTracksWeightDzError) : 0.0f;

  return SharedTracksAndFractions(nSharedTracks, sharedTracksFraction, sharedPt2Fraction, sharedDzErrFraction);
}

// =============================================================================
// reco::VertexCompositePtrCandidate overloads — multi-collection
// =============================================================================

SharedTracksAndFractions calculateVertexSharedTracks(const reco::VertexCompositePtrCandidate &recoV,
                                                     const TrackingVertex &simV,
                                                     const RecoToSimCollectionVec &trackRecoToSimAssociations) {
  unsigned int nSharedTracks = 0;
  float sharedTracksWeightPtSum2 = 0;
  float totalTracksWeightPtSum2 = 0;
  float sharedTracksWeightDzError = 0;
  float totalTracksWeightDzError = 0;

  // Collect reco tracks once; neutral daughters are excluded.
  const std::vector<reco::TrackRef> recoTracks = extractTrackRefs(recoV);

  for (const reco::TrackRef &trkRef : recoTracks) {
    totalTracksWeightPtSum2 += trkRef->pt() * trkRef->pt();
    totalTracksWeightDzError += 1.0 / (trkRef->dzError() * trkRef->dzError());

    // Search across all provided maps — covers the case where PF candidates
    // in the same vertex reference different track collections.
    const auto *tpCandidates = findInRecoToSimCollections(reco::TrackBaseRef(trkRef), trackRecoToSimAssociations);
    if (!tpCandidates)
      continue;

    // Check whether any matched TP is a daughter of the sim vertex.
    for (const auto &tp : *tpCandidates) {
      if (std::find_if(simV.daughterTracks_begin(), simV.daughterTracks_end(), [&](const TrackingParticleRef &vtp) {
            return tp.first == vtp;
          }) != simV.daughterTracks_end()) {
        nSharedTracks += 1;
        sharedTracksWeightPtSum2 += trkRef->pt() * trkRef->pt();
        sharedTracksWeightDzError += 1.0 / (trkRef->dzError() * trkRef->dzError());
        break;
      }
    }
  }

  // Denominator is the number of reco daughters with a recoverable track.
  // Neutral daughters are excluded from the denominator as they carry no
  // tracking information — consistent with their exclusion from the numerator.
  const size_t nRecoTracked = recoTracks.size();
  float sharedTracksFraction = (nRecoTracked > 0) ? (float(nSharedTracks) / float(nRecoTracked)) : 0.0f;
  float sharedPt2Fraction = (totalTracksWeightPtSum2 > 0) ? (sharedTracksWeightPtSum2 / totalTracksWeightPtSum2) : 0.0f;
  float sharedDzErrFraction =
      (totalTracksWeightDzError > 0) ? (sharedTracksWeightDzError / totalTracksWeightDzError) : 0.0f;

  return SharedTracksAndFractions(nSharedTracks, sharedTracksFraction, sharedPt2Fraction, sharedDzErrFraction);
}

SharedTracksAndFractions calculateVertexSharedTracks(const TrackingVertex &simV,
                                                     const reco::VertexCompositePtrCandidate &recoV,
                                                     const SimToRecoCollectionVec &trackSimToRecoAssociations) {
  unsigned int nSharedTracks = 0;
  float sharedTracksWeightPtSum2 = 0;
  float totalTracksWeightPtSum2 = 0;
  float sharedTracksWeightDzError = 0;
  float totalTracksWeightDzError = 0;

  // Collect reco tracks once.
  const std::vector<reco::TrackRef> recoTracks = extractTrackRefs(recoV);

  // Total weights computed from reco side — denominator is always reco tracks.
  for (const reco::TrackRef &trkRef : recoTracks) {
    totalTracksWeightPtSum2 += trkRef->pt() * trkRef->pt();
    totalTracksWeightDzError += 1.0 / (trkRef->dzError() * trkRef->dzError());
  }

  // Iterate from the sim side, mirroring the reco::Vertex overload.
  for (auto iTP = simV.daughterTracks_begin(); iTP != simV.daughterTracks_end(); ++iTP) {
    const auto *tkCandidates = findInSimToRecoCollections(*iTP, trackSimToRecoAssociations);
    if (!tkCandidates)
      continue;

    // Check whether any matched reco track appears in the reco vertex.
    // The reco vertex may mix track collections so compare by (id, key).
    for (const auto &tk : *tkCandidates) {
      if (std::find_if(recoTracks.begin(), recoTracks.end(), [&](const reco::TrackRef &vtk) {
            return (tk.first.id() == vtk.id()) && (tk.first.key() == vtk.key());
          }) != recoTracks.end()) {
        nSharedTracks += 1;
        sharedTracksWeightPtSum2 += tk.first->pt() * tk.first->pt();
        sharedTracksWeightDzError += 1.0 / (tk.first->dzError() * tk.first->dzError());
        break;
      }
    }
  }

  const size_t nRecoTracked = recoTracks.size();
  float sharedTracksFraction = (nRecoTracked > 0) ? (float(nSharedTracks) / float(nRecoTracked)) : 0.0f;
  float sharedPt2Fraction = (totalTracksWeightPtSum2 > 0) ? (sharedTracksWeightPtSum2 / totalTracksWeightPtSum2) : 0.0f;
  float sharedDzErrFraction =
      (totalTracksWeightDzError > 0) ? (sharedTracksWeightDzError / totalTracksWeightDzError) : 0.0f;

  return SharedTracksAndFractions(nSharedTracks, sharedTracksFraction, sharedPt2Fraction, sharedDzErrFraction);
}
