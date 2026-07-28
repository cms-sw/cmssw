#ifndef Validation_RecoVertex_SVEfficiencyEligibility_h
#define Validation_RecoVertex_SVEfficiencyEligibility_h

// =============================================================================
// EfficiencyEligibility
//
// Encodes, per efficiency-cut quantity, whether a SimSecondaryVertex qualifies
// for inclusion in the corresponding efficiency plot — i.e. whether it passes
// all OTHER reconstructability cuts (the variable-blind condition). This
// concept exists exclusively to decide efficiency-plot inclusion; it has no
// bearing on fake rate, duplicate rate, or any other reco-side metric.
//
// Computed in two stages because the PDG ID cut depends on motherPdgId, which
// is expensive to determine (HepMC tree climb) and should only be computed
// for vertices that have a realistic chance of being efficiency-plot eligible:
//
//   Stage 1 (precheckEligibility): evaluates the cheap geometric/multiplicity
//            cuts only (decay length, N daughters) and reports how many
//            of THOSE fail. Used to decide whether motherPdgId is worth
//            computing at all — see needsPdgIdForEfficiency().
//
//   Stage 2 (finalizeEligibility): folds in the PDG ID + pt cuts once
//            motherPdgId and motherPt are known (or defaulted to 0 if Stage 1
//            decided it wasn't worth computing), producing the final per-bundle
//            bitmask.
//
// Stages are implemented in the SecondaryVertexAnalyzerAlgo.
// =============================================================================

/// Bitmask identifying, per efficiency-plot bundle, whether a sim SV is
/// eligible for inclusion — i.e. whether it passes all reconstructability
/// cuts EXCEPT possibly the one matching that bundle's x-axis quantity.
enum class EfficiencyEligibility : uint32_t {
  kNone = 0,
  kDecayLength = 1 << 0,  // eligible for the decay-length efficiency plot
  kNDaughters = 1 << 1,   // eligible for the nTracks efficiency plot
  kPt = 1 << 2,           // eligible for the pt efficiency plot
  kPdgId = 1 << 3,        // eligible for the per-PDG efficiency plots
};

inline EfficiencyEligibility operator&(EfficiencyEligibility a, EfficiencyEligibility b) {
  uint32_t result = static_cast<uint32_t>(a) & static_cast<uint32_t>(b);
  return static_cast<EfficiencyEligibility>(result);
}
inline EfficiencyEligibility operator|(EfficiencyEligibility a, EfficiencyEligibility b) {
  return static_cast<EfficiencyEligibility>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline EfficiencyEligibility& operator|=(EfficiencyEligibility& a, EfficiencyEligibility b) {
  a = a | b;
  return a;
}

// =============================================================================
// Stage 1: cheap-cut precheck
// =============================================================================

/// Result of the cheap-cut precheck (decay length, N daughters, eta only —
/// PDG ID is NOT evaluated here since motherPdgId is not yet known).
struct EfficiencyPrecheck {
  // Bitmask over {kDecayLength, kNDaughters, kPt} only. kPdgId is never
  // set here; it is folded in later by finalizeEligibility().
  EfficiencyEligibility eligibility = EfficiencyEligibility::kNone;

  // Number of cheap cuts (decay length, N daughters, eta) that this vertex
  // fails. Used by potentiallyEligible() to decide whether computing
  // motherPdgId could still make this vertex eligible for some bundle.
  int nFailingCuts = 0;

  /// Returns true if computing motherPdgId is worth doing: i.e. there is at
  /// least one efficiency bundle for which this vertex could still be
  /// eligible once the PDG cut is correctly evaluated.
  ///
  /// Rationale: each bundle suppresses exactly one cut. For the vertex to be
  /// eligible for ANY bundle other than kPdgId itself, it must fail at most
  /// one of the three cheap cuts (the suppressed one) — i.e.
  /// nFailingCuts <= 1. For the kPdgId bundle specifically (which
  /// suppresses the PDG cut but still requires the three cheap cuts to
  /// pass), the vertex must fail NONE of the cheap cuts.
  ///
  /// Hence, the motherPdgId is worth computing whenever nFailingCuts <= 1.
  bool potentiallyEligible() const { return nFailingCuts <= 1; }
};

#endif  // Validation_RecoVertex_SVEfficiencyEligibility_h
