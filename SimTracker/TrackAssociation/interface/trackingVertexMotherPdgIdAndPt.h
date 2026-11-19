#ifndef SimTracker_TrackAssociation_trackingVertexMotherPdgIdAndPt_h
#define SimTracker_TrackAssociation_trackingVertexMotherPdgIdAndPt_h

// Package:    SimTracker/TrackAssociation
//
/**\file trackingVertexMotherPdgIdAndPt.h
   SimTracker/TrackAssociation/interface/trackingVertexMotherPdgIdAndPt.h

 Description: Utility function to determine the PDG ID of the mother particle
              responsible for a given TrackingVertex.

 Motivation:  A TrackingVertex represents a point in the detector where one
              or more particles were produced. Knowing the PDG ID of the
              decaying mother is essential for b/c-tagging validation,
              secondary vertex classification, and any analysis that needs to
              distinguish B-hadron, D-hadron, K-hadron, tau, and other-origin 
              vertices.

              The determination is non-trivial because:

              (a) B and D hadrons are typically handled by the generator and do
                  not themselves become TrackingParticles — only their
                  stable/long-lived descendants do. A naive walk up the
                  TrackingParticle parent chain therefore never reaches them.

              (b) When a TP mother does exist (Case 2 below), it is the most
                  relevant object: a particle that actually propagated in the
                  detector and decayed at this vertex.

              (c) For signal vertices without a TP mother, the mother is a
                  pure generator particle and must be retrieved via the HepMC
                  event record using the G4 track's generator particle index.

              (d) Within the digitisation step, generator vertices within
                  ~10 µm of each other are merged into the same
                  TrackingVertex. The HepMC climb therefore checks that each
                  ancestor vertex is within this merge radius before
                  attributing its PDG ID to the TrackingVertex.

 Cases handled:

   Case 1 — TP mother exists (sourceTracks non-empty):
             The source track(s) of the TrackingVertex are the TPs that decayed
             to produce it. Their PDG IDs are used directly, preferring B > C >
             S > tau > other when multiple source tracks are present.

   Case 2 — No TP mother, pileup vertex (eventId().event() != 0):
             Insufficient MC truth survives for pileup. Returns 0.

   Case 3 — No TP mother, signal vertex, no HepMC event provided:
             Returns 0.

   Case 4 — No TP mother, signal vertex, HepMC event available:
             Climbs the HepMC GenParticle tree via the G4 track's
             genpartIndex(), checking all daughter TPs of the TrackingVertex
             and collecting ancestor PDG IDs from generator vertices within
             the merge radius. Returns the most interesting PDG ID found
             (B > C > S > tau > other non-zero).

 Usage:
   #include "SimTracker/TrackAssociation/interface/trackingVertexMotherPdgIdAndPt.h"

   // With HepMC event (signal vertices get full classification):
   const HepMC::GenEvent* evt = mcProduct.GetEvent();
   int pdgId = sim::trackingVertexMotherPdgIdAndPt(tv, evt);

   // Without HepMC event (pileup and TP-mother cases still handled):
   int pdgId = sim::trackingVertexMotherPdgIdAndPt(tv);

 Note: This header is self-contained and has no .cc counterpart.
       All functions are inline.

 Original Author: Jan Schulz
*/

#include <cmath>
#include <cstdlib>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"

namespace sim {

  // Public interface — forward declarations of helpers used internally
  inline bool isBHadron(int pdgId);
  inline bool isCHadron(int pdgId);
  inline bool isSHadron(int pdgId);
  inline bool isTau(int pdgId);

  namespace detail {
    inline bool isMoreInteresting(int newPdg, int bestSoFar);
    inline bool withinMergeRadius(const HepMC::GenVertex *genVtx, const math::XYZTLorentzVectorD &tvPos);
  }  // namespace detail

  // -------------------------------------------------------------------------
  // Public interface — definitions
  // -------------------------------------------------------------------------

  /// Returns true if pdgId corresponds to a B hadron.
  inline bool isBHadron(const int pdgId) {
    const int absPdg = std::abs(pdgId);
    return (absPdg / 500 == 1)      // B mesons (5xx)
           || (absPdg / 5000 == 1)  // B baryons (5xxx)
           || (absPdg == 5);        // b quark (guard)
  }

  /// Returns true if pdgId corresponds to a charm hadron.
  inline bool isCHadron(const int pdgId) {
    const int absPdg = std::abs(pdgId);
    return (absPdg / 400 == 1)      // D mesons (4xx)
           || (absPdg / 4000 == 1)  // charmed baryons (4xxx)
           || (absPdg == 4);        // c quark (guard)
  }

  /// Returns true if pdgId corresponds to a strange hadron.
  inline bool isSHadron(const int pdgId) {
    const int absPdg = std::abs(pdgId);
    return (absPdg / 300 == 1)      // K mesons (3xx)
           || (absPdg == 130)       // K0_L (130)
           || (absPdg / 3000 == 1)  // strange baryons (4xxx)
           || (absPdg == 4);        // s quark (guard)
  }

  /// Returns true if pdgId corresponds to a tau.
  inline bool isTau(const int pdgId) {
    const int absPdg = std::abs(pdgId);
    return (absPdg == 15);  // tau (15)
  }

  /// Determine the PDG ID of the mother particle responsible for a
  /// TrackingVertex. See file header for a full description of the four
  /// cases handled.
  ///
  /// @param tv        The TrackingVertex to classify.
  /// @param genEvent  Pointer to the HepMC::GenEvent for signal vertices
  ///                  without a TrackingParticle mother. May be nullptr;
  ///                  in that case signal vertices without a TP mother
  ///                  return 0.
  /// @returns         PDG ID of the most physics-relevant ancestor, with
  ///                  B hadrons preferred over D hadrons over K hadrons
  ///                  over tau over others.
  ///                  Returns 0 when classification is not possible.
  inline std::pair<int, double> trackingVertexMotherPdgIdAndPt(const TrackingVertex &tv,
                                                               const HepMC::GenEvent *genEvent = nullptr) {
    using namespace detail;

    if (tv.nDaughterTracks() == 0)
      return {0, 0.};

    // -----------------------------------------------------------------------
    // Case 1: TrackingParticle mother(s) exist — use source tracks directly.
    // sourceTracks() of a TrackingVertex are the TPs that decayed into it.
    // -----------------------------------------------------------------------
    if (tv.nSourceTracks() > 0) {
      int bestPdg = 0;
      double bestPt = 0.0;
      for (auto iTP = tv.sourceTracks_begin(); iTP != tv.sourceTracks_end(); ++iTP) {
        const int pdg = (*iTP)->pdgId();
        if (isMoreInteresting(pdg, bestPdg)) {
          bestPdg = pdg;
          bestPt = (*iTP)->pt();
        }
      }
      return {bestPdg, bestPt};
    }

    // -----------------------------------------------------------------------
    // Case 2: No TP mother, pileup vertex → insufficient MC truth.
    // -----------------------------------------------------------------------
    if (tv.eventId().bunchCrossing() != 0 || tv.eventId().event() != 0)
      return {0, 0.};

    // -----------------------------------------------------------------------
    // Case 3: No TP mother, signal, no HepMC event available.
    // -----------------------------------------------------------------------
    if (!genEvent)
      return {0, 0.};

    // -----------------------------------------------------------------------
    // Case 4: No TP mother, signal vertex → climb HepMC GenParticle tree.
    //
    // For each daughter TP of this TrackingVertex:
    //   - Retrieve its HepMC GenParticle via the G4 track's genpartIndex.
    //   - Walk up the production vertices of that GenParticle.
    //   - For each generator vertex within the merge radius, inspect all
    //     incoming particles and record the most interesting PDG ID found.
    //   - Stop climbing when we leave the merge radius or reach a B hadron.
    // -----------------------------------------------------------------------
    const math::XYZTLorentzVectorD tvPos(tv.position().x(), tv.position().y(), tv.position().z(), tv.position().t());

    int bestPdg = 0;
    double bestPt = 0.0;
    bool foundB = false;

    for (auto iTP = tv.daughterTracks_begin(); iTP != tv.daughterTracks_end() && !foundB; ++iTP) {
      const TrackingParticle &tp = **iTP;

      if (tp.g4Tracks().empty())
        continue;
      const int genIndex = tp.g4Tracks()[0].genpartIndex();
      if (genIndex < 0)
        continue;

      const HepMC::GenParticle *daughterGen = genEvent->barcode_to_particle(genIndex);
      if (!daughterGen)
        continue;

      // Walk up the production vertex chain
      const HepMC::GenVertex *prodVtx = daughterGen->production_vertex();
      while (prodVtx && !foundB) {
        // Stop if this generator vertex is outside the merge radius
        if (!withinMergeRadius(prodVtx, tvPos))
          break;

        // Inspect all incoming particles at this generator vertex
        for (auto iMother = prodVtx->particles_in_const_begin(); iMother != prodVtx->particles_in_const_end();
             ++iMother) {
          const int pdg = (*iMother)->pdg_id();

          if (isMoreInteresting(pdg, bestPdg)) {
            bestPdg = pdg;
            bestPt = (*iMother)->momentum().perp();
          }

          // B hadron is the most interesting possible — stop everything
          if (isBHadron(pdg)) {
            foundB = true;
            break;
          }
        }

        // Continue climbing via the first incoming particle's production vtx
        if (!foundB && prodVtx->particles_in_size() > 0)
          prodVtx = (*prodVtx->particles_in_const_begin())->production_vertex();
        else
          break;
      }
    }

    return {bestPdg, bestPt};
  }

  // -------------------------------------------------------------------------
  // Internal helpers — definitions
  // -------------------------------------------------------------------------

  namespace detail {
    /// Returns true if newPdg is more physics-relevant than bestSoFar
    /// for secondary vertex classification purposes (B > C > S > tau > other).
    inline bool isMoreInteresting(int newPdg, int bestSoFar) {
      if (isBHadron(newPdg) && !isBHadron(bestSoFar))
        return true;
      if (isCHadron(newPdg) && !isBHadron(bestSoFar) && !isCHadron(bestSoFar))
        return true;
      if (isSHadron(newPdg) && !isBHadron(bestSoFar) && !isCHadron(bestSoFar) && !isSHadron(bestSoFar))
        return true;
      if (isTau(newPdg) && !isBHadron(bestSoFar) && !isCHadron(bestSoFar) && !isSHadron(bestSoFar) && !isTau(bestSoFar))
        return true;
      if (bestSoFar == 0 && newPdg != 0)
        return true;
      return false;
    }

    /// Maximum 3D distance [cm] within which generator vertices are merged
    /// into a single TrackingVertex during digitisation (~10 µm).
    /// link: https://github.com/cms-sw/cmssw/blob/140d0f0f1f2ee369bc8996185c9399384005bab6/SimG4Core/Notification/src/SimTrackManager.cc#L204
    constexpr double kMergeRadius = 0.001;  // cm

    /// Returns true if the HepMC generator vertex genVtx is within
    /// kMergeRadius of the TrackingVertex position tvPos [cm].
    /// HepMC stores positions in mm; conversion to cm is applied here.
    inline bool withinMergeRadius(const HepMC::GenVertex *genVtx, const math::XYZTLorentzVectorD &tvPos) {
      if (!genVtx)
        return false;
      const HepMC::ThreeVector &gp = genVtx->point3d();
      const double dx = gp.x() * 0.1 - tvPos.x();  // mm → cm
      const double dy = gp.y() * 0.1 - tvPos.y();
      const double dz = gp.z() * 0.1 - tvPos.z();
      return (dx * dx + dy * dy + dz * dz) < kMergeRadius * kMergeRadius;
    }
  }  // namespace detail

}  // namespace sim

#endif  // SimTracker_TrackAssociation_trackingVertexMotherPdgIdAndPt_h
