#ifndef Validation_RecoVertex_SecondaryVertexAnalyzerAlgo_h
#define Validation_RecoVertex_SecondaryVertexAnalyzerAlgo_h

// Package:    Validation/RecoVertex
// Class:      SecondaryVertexAnalyzerAlgo
//
/**\class SecondaryVertexAnalyzerAlgo
   Validation/RecoVertex/interface/SecondaryVertexAnalyzerAlgo.h

 Description: Algorithm class for secondary vertex validation. Owns all
              histogram booking and filling logic, completely decoupled from
              the EDM framework. Receives already-fetched collections and
              association maps from the plugin (SecondaryVertexAnalyzer.cc).

 Deliberately contains no EDM includes. All framework interaction is the
 responsibility of the plugin.

 Sim vertex handling philosophy
 ───────────────────────────────
 Two distinct sim vertex collections are maintained per event, mirroring the
 approach used in track validation:

   allSimSVs_     All non-PV TrackingVertices, with at least one charged 
                  daughter particle of a certain min pT but no physics selection
                  applied. Used as the truth reference for fake rate, duplicate
                  rate, and pileup rate estimates, so that these rates reflect
                  the full landscape of true secondary vertices in the event.

   signalSimSVs_  Subset of allSimSVs_ passing the configured signal selection
                  (minReconstructableDaughters, PDG ID filter if set). 
                  Used as the denominator for efficiency estimates.

 Variable-blind cut suppression
 ───────────────────────────────
 Efficiency plots whose x-axis is the same quantity as a selection cut must
 NOT apply that cut, otherwise the efficiency is trivially 1 or 0 at the
 boundaries. This is handled via the ReconstructabilityFlags bitmask: each
 monitoring bundle is associated with a set of flags that identifies which
 cut(s) to suppress when evaluating reconstructability for that bundle's
 sim-side fills. The isEligibleForEff() predicate accepts a EfficiencyEligibility 
 value and bypasses the corresponding checks.

 Example:
   h_decayLength bundle  → SkipCuts::kDecayLength  (do not apply minDecayLength/maxDecayLength)
   h_nTracks bundle      → SkipCuts::kNDaughters   (do not apply minReconstructableDaughters)
   all other bundles     → SkipCuts::kNone         (apply all cuts)

 Original Author: Jan Schulz
*/

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// DQM
#include "DQMServices/Core/interface/DQMStore.h"

// Vertex data formats
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

// Sim truth
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

// Internal types and bundles
#include "Validation/RecoVertex/interface/SVMonitoringBundle.h"
#include "Validation/RecoVertex/interface/SVResolutionBundle.h"
#include "Validation/RecoVertex/interface/SVTrackQualityBundle.h"
#include "Validation/RecoVertex/interface/SVValidationStructs.h"

class SecondaryVertexAnalyzerAlgo {
public:
  using IBooker = dqm::reco::DQMStore::IBooker;
  using AssociatorVtx = reco::VertexToTrackingVertexAssociator<std::vector<reco::Vertex>>;
  using RecoToSimCollectionVtx = AssociatorVtx::RecoToSimCollection;
  using SimToRecoCollectionVtx = AssociatorVtx::SimToRecoCollection;
  using AssociatorCPC = reco::VertexToTrackingVertexAssociator<std::vector<reco::VertexCompositePtrCandidate>>;
  using RecoToSimCollectionCPC = AssociatorCPC::RecoToSimCollection;
  using SimToRecoCollectionCPC = AssociatorCPC::SimToRecoCollection;

  // =========================================================================
  // Configuration
  // =========================================================================

  /// Configuration struct — populated from the ParameterSet by the plugin
  /// and passed here so the algo has no PSet dependency.
  struct Config {
    std::string rootFolder;
    bool verbose;
    bool doGenericSimPlots;  // book/fill collection-independent sim plots
    bool doPerPdgPlots;      // book per-b/c/other efficiency breakdowns

    // Signal selection cuts applied to build signalSimSVs_.
    // Each cut is individually suppressed for the monitoring bundle
    // whose x-axis is that quantity (see SkipCuts below).
    double minDecayLength;                                    // minimum 3D decay length [cm]
    double maxDecayLength;                                    // maximum 3D decay length [cm]
    double minPt;                                             // minimum pT of Sim SV (vector sum of charged daughters)
    double minPtReconstructableDaughters;                     // minimum pT of charged daughters
    int minReconstructableDaughters;                          // minimum charged daughters
    bool bHadrons, cHadrons, sHadrons, taus, otherParticles;  // include certain types for eff

    // Optional PDG ID filter: if non-empty, only sim SVs whose mother PDG ID
    // (absolute value) appears in this list are included in signalSimSVs_.
    // Empty means no PDG filter (accept all).
    std::vector<int> signalPdgIds;
  };

  // =========================================================================
  // Reconstructability cut suppression
  // =========================================================================

  /// Bitmask `EfficiencyEligibility` used to implement variable-blind
  /// efficiency plots: the bundle whose x-axis is quantity X suppresses the
  /// cut on X when evaluating reconstructability for its sim-side fills.
  using EffElig = EfficiencyEligibility;

  /// Returns true if sv passes the reconstructability criteria, optionally
  /// suppressing individual cuts as indicated by mask.
  bool isEligibleForEff(const SimSecondaryVertex &sv, EfficiencyEligibility mask = EffElig::kNone) const;

  // =========================================================================
  // Public interface
  // =========================================================================

  explicit SecondaryVertexAnalyzerAlgo(const Config &cfg);
  ~SecondaryVertexAnalyzerAlgo() = default;

  /// Called from DQMEDAnalyzer::bookHistograms.
  void bookHistograms(IBooker &ibook, const std::vector<std::string> &collectionLabels);

  /// Per-event entry point for reco::Vertex collections (track-based SVs).
  void analyze(const edm::View<reco::Vertex> &recoVertices,
               const RecoToSimCollectionVtx &recoToSim,
               const SimToRecoCollectionVtx &simToReco,
               const reco::SimToRecoCollection &trackSimToReco,
               const std::string &collectionLabel);

  /// Per-event entry point for reco::VertexCompositePtrCandidate collections.
  void analyze(const edm::View<reco::VertexCompositePtrCandidate> &recoVertices,
               const RecoToSimCollectionCPC &recoToSim,
               const SimToRecoCollectionCPC &simToReco,
               const reco::SimToRecoCollection &trackSimToReco,
               const std::string &collectionLabel);

  /// Set the (reco) primary vertex of the event once since it's needed for decay length calculation.
  /// If PV collection is unavailable, default to the detector center.
  void setPrimaryVertex(const edm::Handle<reco::VertexCollection> &pvsHandle);

  /// Build both the full sim SV list and a reference list for the subset used for efficiency calculation.
  /// Exposed as a public function so it be called only once for all reco SV collections together.
  void prepareEventTruth(const edm::Handle<TrackingVertexCollection> &simVerticesH, const HepMC::GenEvent *genEvent);

  /// Clear the SimVertex SVs at the end of the event.
  void clearEventTruth();

  /// Fill generic SimVertex histograms (collection-independent, filled once per event).
  void fillEventTruthHistograms() { fillGenericSimVertexHistograms(); };

private:
  // =========================================================================
  // Sim vertex building
  // =========================================================================

  /// Build the full sim SV list: all non-PV TrackingVertices with decay
  /// length and mother PDG ID populated. No signal selection applied.
  /// Used as truth reference for fake/duplicate/pileup rate estimates.
  std::vector<SimSecondaryVertex> buildAllSimSVs(const edm::Handle<TrackingVertexCollection> &simVerticesH) const;

  /// Apply signal selection to allSimSVs_ to produce the efficiency
  /// denominator. Applies minDecayLength, minReconstructableDaughters,
  /// and signalPdgIds from Config.
  std::vector<SimSecondaryVertex *> buildSignalSimSVs(const HepMC::GenEvent *genEvent);

  /// Reset all the reco-matching-dependent information in the SimVertex SVs.
  /// Should be called everytime a new RecoVertex collection is analyzed.
  void resetSimSVs();

  /// Compute 3D decay length w.r.t. the hard-scatter primary vertex.
  double decayLength(const TrackingVertex &tv, const TrackingVertex &pv, const bool decayLength2D) const;

  /// Stage 1 of setting the EfficiencyEligibility flags:
  /// evaluate the cheap reconstructability cuts (decay length,
  /// N daughters, eta) and report per-cut pass/fail plus the eligibility
  /// bitmask restricted to those three cuts.
  EfficiencyPrecheck precheckEligibility(const SimSecondaryVertex &sv) const;

  /// Stage 2: given the Stage 1 precheck and the now-known motherPdgId (or 0
  /// if Stage 1 decided it wasn't worth computing), produce the final
  /// per-bundle eligibility bitmask, including kPdgId.
  /// Returns true, if eligible for some efficiency plot.
  bool finalizeEligibility(SimSecondaryVertex &sv, const EfficiencyPrecheck &precheck) const;
  // =========================================================================
  // Reco vertex building
  // =========================================================================

  std::vector<RecoSecondaryVertex> buildRecoSVs(const edm::View<reco::Vertex> &recoVertices) const;

  std::vector<RecoSecondaryVertex> buildRecoSVs(const edm::View<reco::VertexCompositePtrCandidate> &recoVertices) const;

  // =========================================================================
  // Association and matching
  // =========================================================================

  /// Sim→Reco direction: populates SimSecondaryVertex::nMatchedRecoVertices
  /// and matchedQualities. Operates on allSimSVs so that all
  /// true SVs, including pileup, are considered for matching.
  template <typename AssociatorType>
  void matchSim2RecoVertices(const AssociatorType &simToReco);

  /// Reco→Sim direction: populates RecoSecondaryVertex::kind_of_vertex,
  /// simVertices, and matchedQualities.
  ///
  /// The full allSimSVs collection is used here so that:
  ///   - A reco SV matched only to a pileup sim SV is correctly flagged
  ///     as pileup rather than fake.
  ///   - A reco SV matched to multiple sim SVs (merged) is correctly
  ///     identified even if only one of those sim SVs passes signal selection.
  ///
  /// The signalSimSVs pointer set is used only to distinguish, for matched
  /// reco SVs, whether the matched sim SV is a signal vertex or not.
  template <typename AssociatorType>
  void matchReco2SimVertices(std::vector<RecoSecondaryVertex> &recoSVs, const AssociatorType &recoToSim) const;

  /// For each signal sim SV, counts how many of its charged daughter
  /// TrackingParticles have at least one associated reconstructed track (via
  /// the track-level SimToReco association map). If at least two daughters have
  /// a matched track, the vertex is marked reconstructable.
  void setSignalSimSVReconstructability(const reco::SimToRecoCollection &trackSimToReco);

  // =========================================================================
  // Histogram filling
  // =========================================================================

  /// Fill generic sim-side histograms for all SimSecondaryVertices.
  /// Fill collection independent histograms once.
  void fillGenericSimVertexHistograms();

  /// Fill sim-side histograms for one SimSecondaryVertex.
  /// Each bundle is filled with its associated SkipCuts mask applied to the
  /// reconstructability evaluation — this is the variable-blind mechanism.
  void fillSimVertexHistograms(const std::string &label, const SimSecondaryVertex &sv);

  /// Fill reco-side histograms for one RecoSecondaryVertex.
  void fillRecoVertexHistograms(const std::string &label, const RecoSecondaryVertex &rv);

  /// Fill resolution/pull histograms for a matched reco-sim pair.
  void fillResolutionHistograms(const std::string &label, const RecoSecondaryVertex &rv, const SimSecondaryVertex &sv);
  void fillTrackQualityHistograms(const std::string &label,
                                  const RecoSecondaryVertex &rv,
                                  const SimSecondaryVertex &sv);

  // =========================================================================
  // Shared implementation
  // =========================================================================

  /// Internal template called by both public analyze() overloads after
  /// type-specific buildRecoSVs(), buildAllSimSVs() and matchReco2SimVertices() have been called.
  template <typename SimToRecoAssociationType, typename RecoToSimAssociationType>
  void analyzeImpl(std::vector<RecoSecondaryVertex> recoSVs,
                   const RecoToSimAssociationType &recoToSim,
                   const SimToRecoAssociationType &simToReco,
                   const reco::SimToRecoCollection &trackSimToReco,
                   const std::string &collectionLabel);

  // =========================================================================
  // Histogram storage
  // =========================================================================

  // Per-collection bundle structs. Each SVMonitoringBundle is associated
  // with a mask that is applied when evaluating the eligibility for efficiency calculation
  // during sim-side fills — this implements variable-blind cut suppression.
  struct BundleWithCutMask {
    SVMonitoringBundle bundle{};
    EfficiencyEligibility mask = EffElig::kNone;  // which eligibility to check
  };

  struct CollectionHistograms {
    // Efficiency / fake rate monitoring bundles.
    // mask encodes which cut is suppressed for each bundle's x-axis.
    BundleWithCutMask h_decayLength{.mask = EffElig::kDecayLength};
    BundleWithCutMask h_decayLengthSig{.mask = EffElig::kDecayLength};
    BundleWithCutMask h_decayLengthXY{.mask = EffElig::kDecayLength};
    BundleWithCutMask h_decayLengthXYSig{.mask = EffElig::kDecayLength};
    BundleWithCutMask h_r{};
    BundleWithCutMask h_nTracks{.mask = EffElig::kNDaughters};
    BundleWithCutMask h_eta{};
    BundleWithCutMask h_chi2ndof{};
    BundleWithCutMask h_pt{.mask = EffElig::kPt};
    BundleWithCutMask h_mass{};

    // Resolution bundles — filled only for matched reco-sim pairs,
    // no reconstructability evaluation needed here.
    SVResolutionBundle h_xRes;
    SVResolutionBundle h_yRes;
    SVResolutionBundle h_zRes;
    SVResolutionBundle h_decayLengthRes;
    SVResolutionBundle h_decayLengthXYRes;
    SVResolutionBundle h_ptRes;
    SVResolutionBundle h_etaRes;
    SVResolutionBundle h_phiRes;
    SVResolutionBundle h_massRes;

    // Track content quality bundle — filled only for matched signal reco-sim pairs
    SVTrackQualityBundle h_trackQuality;
    SVTrackQualityBundle h_trackQuality_nTracksSimSV;
    SVTrackQualityBundle h_trackQuality_nTracksRecoSV;
    SVTrackQualityBundle h_trackQuality_decayLength;
    SVTrackQualityBundle h_trackQuality_decayLengthXY;
    SVTrackQualityBundle h_trackQuality_chi2ndof;

    // Additional MonitorElements keyed by histogram name
    std::map<std::string, dqm::reco::MonitorElement *> mes;
  };

  // Generic sim-side histograms booked once (collection-independent).
  // Only populated when cfg_.doGenericSimPlots is true.
  struct GenericSimHistograms {
    dqm::reco::MonitorElement *h_decayLength = nullptr;
    dqm::reco::MonitorElement *h_decayLengthXY = nullptr;
    dqm::reco::MonitorElement *h_nDaughters = nullptr;
    dqm::reco::MonitorElement *h_motherPdgId = nullptr;
    dqm::reco::MonitorElement *h_numAllSimSVs = nullptr;
    dqm::reco::MonitorElement *h_numSignalSimSVs = nullptr;
  };

  // =========================================================================
  // Class members
  // =========================================================================

  const Config cfg_;

  // MonitorElements and bundles keyed by [collectionLabel].
  std::map<std::string, CollectionHistograms> collectionHistos_;
  GenericSimHistograms genericSimHistos_;

  // MC truth SimVertices for SVs (built once for all collections together and then reused)
  std::vector<SimSecondaryVertex> allSimSVs_;
  std::vector<SimSecondaryVertex *> signalSimSVs_;

  // Primary vertex (reco) of the event, defaults to CMS center if unavailable
  reco::Vertex pv_;
};

#endif  // Validation_RecoVertex_SecondaryVertexAnalyzerAlgo_h
