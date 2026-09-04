// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Histogram definitions for the truth-branch validation, kept apart from the analyzer
// the way MTVHistoProducerAlgoForTracker and HGVHistoProducerAlgo are: a POD struct
// that owns only MonitorElement pointers, plus a stateless algorithm whose fill_*
// methods take that struct by CONST reference. That is what lets the analyzer be a
// DQMGlobalEDAnalyzer, where booking and filling are both const and the MEs live in a
// per-run cache.
//
// Only num/denom histograms are booked. Every efficiency, fake rate and duplicate rate
// is formed downstream by DQMGenericClient from the string configuration, so this
// package contains no harvesting C++ at all.
//
// The truth side and the reco side are binned in DIFFERENT variable sets, because they
// describe different objects. Efficiency and duplicate rate divide two truth-side
// histograms and are binned in branch variables, which every domain supplies. Purity,
// fake rate and pileup rate divide two reco-side histograms and are binned in the reco
// object's own variables, which a vertex and a trackster do not share with a track.
// Booking a variable a domain cannot fill would put a spike at zero into every such
// plot and read as a real feature.

#ifndef Validation_TruthInfo_TruthBranchHistoProducerAlgo_h
#define Validation_TruthInfo_TruthBranchHistoProducerAlgo_h

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"

namespace truth {

  // Acceptance regions. Every num_* row is booked once inclusively and once per region,
  // in a sub-folder of the same name, so a metric can be read where the detector that
  // reconstructs it actually is. Mixing them is not a detail: on no-PU TenTau 58.6% of
  // the taus enter the calorimeter in the barrel, where no trackster can exist, which
  // pulled the inclusive calorimetric efficiency from 0.49 down to 0.19.
  //
  // Index 0 is the inclusive row and is always filled. An object outside every band, or
  // one whose region variable is undefined, is filled ONLY there.
  // Bands in ABSOLUTE pseudorapidity, so every metric is measured once per band and the
  // two detector halves are pooled. Splitting the endcaps by sign would multiply every
  // plot in every domain by 1.5 to serve the handful where the two lobes are physically
  // far apart, which is a question about drawing a position axis and not about acceptance.
  enum class EtaRegion { Inclusive = 0, Barrel, Endcap, Forward };
  inline constexpr std::size_t kNEtaRegions = 4;
  inline static const std::vector<std::string> kEtaRegionFolders = {"", "etaLt15", "eta15to30", "eta30to45"};

  // Which band an absolute pseudorapidity falls in; Inclusive when it is in none.
  [[nodiscard]] inline EtaRegion etaRegionOf(double absEta) {
    if (absEta < 1.5)
      return EtaRegion::Barrel;
    if (absEta < 3.0)
      return EtaRegion::Endcap;
    if (absEta < 4.5)
      return EtaRegion::Forward;
    return EtaRegion::Inclusive;
  }

  // The x variables, following the MTVHistoProducerAlgoForTracker set restricted to
  // what a truth branch can supply, plus two the graph alone can supply: depth is how
  // far down the event history the branch root sits, and root_footprint_fraction is how much of the
  // branch footprint belongs to the root particle itself rather than to its
  // descendants.
  enum class Variable { Pt, Eta, Phi, Nhits, Vertpos, Zpos, Dxy, Dz, Depth, RootFootprintFraction, CaloEta, Flavour };
  inline static const std::vector<std::string> kVariableNames = {"pt",
                                                                 "eta",
                                                                 "phi",
                                                                 "nhits",
                                                                 "vertpos",
                                                                 "zpos",
                                                                 "dxy",
                                                                 "dz",
                                                                 "depth",
                                                                 "root_footprint_fraction",
                                                                 "caloeta",
                                                                 "flavour"};

  // The species that initiated a truth object, as a bin index. Answers "what kind of
  // particle made this" on the same axis machinery every other variable uses, so
  // efficiency and efficiency_cumulative against it need no special case.
  //
  // Only the partonJets level populates anything but Other: every level is booked on
  // every axis, and a level whose roots are not partons is entirely in bin 0 by
  // construction rather than by accident.
  enum class FlavourBin { Other = 0, Down, Up, Strange, Charm, Bottom, Top, Gluon };
  inline static constexpr int kNFlavourBins = 8;
  inline static const std::vector<std::string> kFlavourBinNames = {"other", "d", "u", "s", "c", "b", "t", "g"};

  [[nodiscard]] inline double flavourBin(int32_t pdgId) {
    const int32_t a = std::abs(pdgId);
    if (a >= 1 && a <= 6)
      return static_cast<double>(a) + 0.5;
    if (a == 21)
      return static_cast<double>(FlavourBin::Gluon) + 0.5;
    return static_cast<double>(FlavourBin::Other) + 0.5;
  }

  // caloeta of a branch that never reached the calorimeter. Far outside every axis
  // range, so such a branch lands in the underflow of BOTH numerator and denominator
  // and the calorimeter-entrance axis shows only what a calorimeter could have seen.
  inline constexpr double kNoCaloEntry = -999.;

  struct TruthBranchHistograms {
    using METype = dqm::reco::MonitorElement*;

    // Rows are booked in blocks of kNEtaRegions per entry: index kNEtaRegions * entry + r
    // is entry's row for region r, r = 0 being inclusive. The fill side does that
    // arithmetic once and fills exactly two rows, the inclusive one and the object's.
    //
    // Each vector is indexed [entry][variable], variable being the position within that
    // side's variable list, so booking order and fill index stay in step exactly as in
    // MTV. The two sides carry INDEPENDENT entry counters: truth-driven rows are
    // indexed by (collection, level) in truth-entry booking order, reco-driven rows by
    // (collection, working point) in wp-entry booking order.
    using MERow = std::vector<METype>;

    // Truth side, one row per (collection, level): denominator every target at that
    // level, numerator those a reco object was associated to. The cumulative numerator
    // also accepts targets covered only by several reco objects together, so it is a
    // superset of the individual one by construction.
    std::vector<MERow> h_simul, h_assoc_simToReco, h_assoc_simToReco_cumulative;

    // The two ways a truth object can be reconstructed as one object more than once or
    // in pieces, mutually exclusive so that individual + duplicate + split + lost = 1.
    //   duplicate  more than one reco object individually reconstructs the whole thing
    //   split      no single object does, but several together cover the subgraph
    // h_duplicate is left EMPTY for a calorimetric domain, where the outcome cannot
    // occur: two reco objects built from disjoint layer clusters cannot each miss less
    // than maxSimToRecoScoreForDuplicate of the same branch energy, since the two scores
    // sum to at least one. Measured on 200 no-PU ttbar events: ticlCandidate,
    // ticlTrackstersCLUE3DHigh and ticlTracksterLinks each use every layer cluster in at
    // most one trackster. A collection whose objects SHARE hits would make it reachable
    // again and would have to book it. Split carries the calorimetric pathology instead.
    std::vector<MERow> h_duplicate, h_split;

    // Reco side, one row per (collection, working point): denominator every reco
    // object, and three numerators answering three different questions. Pileup counts
    // objects matched only to an overlaid interaction.
    //
    // h_dominated is the FAKE numerator: the object matched something AND, where the
    // dominance question is defined for it, one truth branch of the antichain owns at
    // least minLeadingTruthShare of the shared quantity. A fake is an object matched to
    // nothing, or one whose contributions are comparably small with no winner.
    //
    // An object that matched truth but has NO candidate at the dominance level is not a
    // fake. The question is undefined for it, not answered negatively, and folding it in
    // measures level coverage rather than reconstruction: on no-PU ttbar it is 32.5% of
    // tracksters and 36.8% of tracks, against 0.3% of tracks matched to nothing.
    // h_levelCandidate counts the objects where the question IS defined, so the
    // complement is published as its own page and stays visible.
    //
    // h_assoc_recoToSim counts objects matched to anything, one entry each, published as
    // the no-candidate rate and named for what it measures so it cannot be read as a
    // second fake rate.
    //
    // h_recopurity fills matched objects weighted by the purity of the match, so its
    // ratio to h_reco is the mean purity. It must stay separate from the counts:
    // filling one histogram with the purity as a weight and reading it as a count turns
    // the fake rate into one minus the mean purity, which on no-PU ttbar reads 0.83
    // where the fake rate is 0.003.
    //
    // h_assoc_strict is the calorimetric domains' numerator for HGCalValidator's
    // non-fake criterion, matched AND below maxRecoToSimScore, kept only so the two
    // validators stay comparable. That criterion is not a fake rate: it is normalised
    // against the cell's total truth energy, so pileup on a cell drives it towards 1
    // even for a good match. Booked for calorimetric domains only; empty elsewhere.
    std::vector<MERow> h_reco, h_dominated, h_levelCandidate, h_assoc_recoToSim, h_recopurity, h_pileup, h_assoc_strict;

    // Efficiency and duplicate rate against the Geant4 process that CREATED the
    // branch, which only the graph can supply: the production vertex of the branch
    // root carries its VertexReason, so a loss can be attributed to the physics that
    // made the particle rather than only to where it landed. Truth side.
    std::vector<METype> h_simul_reason, h_assoc_simToReco_reason, h_duplicate_reason;

    // Quality of the match itself, one per direction. The denominator is what the name
    // says: reco purity divides by the reco object (reco side), truth purity by the
    // truth object (truth side).
    std::vector<METype> h_score, h_sharedQuantity, h_recoPurity, h_truthPurity;

    // DOMINANCE of the leading truth contributor, the axis a fake criterion built on
    // "no truth dominates the little contaminations" would cut on. leading_truth_share
    // is the leading branch's shared energy over the shared energy of ALL candidate
    // branches; dominance_ratio is leading over runner-up, capped at 20. Reco side, and
    // both are read from the FIRST working point's map, the only one that carries every
    // candidate. Filled for every reco object with at least one candidate.
    std::vector<METype> h_leadingShare, h_dominanceRatio;

    // The axis the calorimetric efficiency cut acts on: shared energy over the truth
    // branch's own energy. Booked only by the domains judged on it, so it is empty for
    // every other one. Truth side.
    std::vector<METype> h_sharedEnergyFraction;

    // Resolution inputs: 2D of (reco - truth)/truth against the truth variable, which
    // the harvester turns into _Mean and _Sigma by a Gaussian fit per slice. Reco side:
    // the pair comes from the reco-driven match, so it depends on the working point.
    std::vector<METype> h_ptres_vs_eta, h_ptres_vs_pt, h_etares_vs_eta, h_phires_vs_eta;
  };

  class TruthBranchHistoProducerAlgo {
  public:
    explicit TruthBranchHistoProducerAlgo(edm::ParameterSet const& pset);

    // Book one set of histograms into the current folder, appending one row to each of
    // that side's vectors. Call bookRecoHistos once per (collection, working point) and
    // bookTruthHistos once per (collection, level), each in the order the fill side
    // will index that list.
    // calorimetric additionally books the strict numerator described above.
    void bookRecoHistos(dqm::implementation::IBooker& booker,
                        TruthBranchHistograms& histograms,
                        bool calorimetric) const;
    // calorimetric books the shared-energy-fraction monitor element, the axis those
    // domains gate efficiency on, and skips the duplicate ones the same domains cannot
    // fill. It must be the same for every truth entry of one module, so the row index
    // stays shared with the other truth vectors.
    void bookTruthHistos(dqm::implementation::IBooker& booker,
                         TruthBranchHistograms& histograms,
                         bool calorimetric) const;

    // One region's worth of rows, into the booker's current folder. bookTruthHistos and
    // bookRecoHistos call these once per region.
    void bookTruthRow(dqm::implementation::IBooker& booker, TruthBranchHistograms& histograms, bool calorimetric) const;
    void bookRecoRow(dqm::implementation::IBooker& booker, TruthBranchHistograms& histograms, bool calorimetric) const;

    // The once-per-entry distributions, booked in the base folder only.
    void bookTruthDiagnostics(dqm::implementation::IBooker& booker,
                              TruthBranchHistograms& histograms,
                              bool calorimetric) const;
    void bookRecoDiagnostics(dqm::implementation::IBooker& booker,
                             TruthBranchHistograms& histograms,
                             bool calorimetric) const;

    // Values of every x variable for one object, in the enum order. A domain fills only
    // the ones it has; which of them are booked is decided by the variable lists.
    struct Kinematics {
      double pt = 0., eta = 0., phi = 0., nhits = 0., vertpos = 0., zpos = 0., dxy = 0., dz = 0.;
      double depth = 0., root_footprint_fraction = 0., caloeta = kNoCaloEntry;
      double flavour = static_cast<double>(FlavourBin::Other) + 0.5;
      std::array<double, 12> asVector() const {
        return {pt, eta, phi, nhits, vertpos, zpos, dxy, dz, depth, root_footprint_fraction, caloeta, flavour};
      }
    };

    // How one truth object was reconstructed. Exactly one of these is true.
    enum class TruthOutcome { Individual, Duplicate, Split, Lost };

    // cumulative is true when the collection as a whole covers the truth object,
    // whether by one reco object or by several together.
    // The two row-level fills, one region's row each. fill_simul and fill_reco call them
    // for the inclusive row and for the object's region row.
    // failedCuts is a BranchSelector::CutBit mask of the plotted-axis cuts this object
    // fails. A variable is filled only when the object fails nothing except the cut on
    // that variable itself, so an efficiency against pt keeps the objects the pt cut
    // would have removed and no other plot is polluted by them.
    void fill_simul_row(TruthBranchHistograms const& histograms,
                        std::size_t index,
                        Kinematics const& kin,
                        TruthOutcome outcome,
                        bool cumulative,
                        uint32_t failedCuts) const;

    void fill_simul(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    Kinematics const& kin,
                    TruthOutcome outcome,
                    bool cumulative,
                    uint32_t failedCuts) const;

    // linthresh > 0 asks for SYMLOG binning: one linear bin [min, linthresh] and the rest
    // log-spaced up to max. Plain log cannot represent 0, and on DY 20.5% of the signal
    // level sits at pt EXACTLY 0, the pre-ISR copy of the resonance, so a log axis would
    // move a fifth of that denominator into the underflow where nobody would see it.
    //
    // float rather than double because that is what the DQM booker's variable-bin
    // overload takes; histogram edges do not need more.
    struct SymlogAxis {
      int nbins;
      double min, max, linthresh;
    };
    [[nodiscard]] static std::vector<float> binEdges(SymlogAxis const& axis);

    // The cut bit a truth variable is the axis of, or 0 for a variable no cut touches.
    // Indexed like kVariableNames, so booking order and this table cannot drift.
    [[nodiscard]] static uint32_t cutBitOfVariable(std::string const& name);

    // Truth purity of the leading reco object, filled once per truth object that has
    // any overlap at all.
    void fill_truth_purity(TruthBranchHistograms const& histograms, std::size_t index, double truthPurity) const;

    // Shared energy fraction of the leading reco object, filled once per truth object
    // that has any overlap at all, by the domains that booked it.
    void fill_shared_energy_fraction(TruthBranchHistograms const& histograms,
                                     std::size_t index,
                                     double sharedEnergyFraction) const;

    // How one reco object relates to the truth. Grouped into a struct rather than
    // passed as five positional flags, which no call site can get right by inspection.
    struct RecoOutcome {
      // Not a fake: matched, and not contaminated beyond attribution. Its complement is
      // the fake rate.
      bool dominated = false;
      // Matched to anything at all, one of the two ways of being a fake on its own.
      bool associated = false;
      // The dominance question is DEFINED for this object, that is at least one candidate
      // projects onto the antichain. Its complement is published as its own page and is
      // deliberately not a fake.
      bool hasLevelCandidate = false;
      bool pileup = false;
      // Calorimetric only, HGCalValidator's non-fake criterion. Fills h_assoc_strict
      // and nothing else, so it can never move the fake rate.
      bool strictMatch = false;
      // Purity of the match: 1 minus the reco-normalised score for a hit-based domain,
      // the leading truth vertex's share of the constituents for a composite one. It
      // weights the h_recopurity fill only; every other fill here is a count.
      double matchQuality = 1.;
    };

    void fill_reco(TruthBranchHistograms const& histograms,
                   std::size_t index,
                   Kinematics const& kin,
                   RecoOutcome const& outcome) const;

    void fill_reco_row(TruthBranchHistograms const& histograms,
                       std::size_t index,
                       Kinematics const& kin,
                       RecoOutcome const& outcome) const;

    // Categorical fill against the VertexReason of the branch root's production
    // vertex, passed as its underlying integer so this header stays free of the
    // graph data formats.
    void fill_reason(TruthBranchHistograms const& histograms,
                     std::size_t index,
                     unsigned int reason,
                     TruthOutcome outcome) const;

    // Negative values mean the object had no candidate at all and are not filled.
    void fill_dominance(TruthBranchHistograms const& histograms,
                        std::size_t index,
                        double leadingShare,
                        double dominanceRatio) const;

    void fill_match(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    double score,
                    double sharedQuantity,
                    double recoPurity) const;

    // Called once per matched pair, with the truth branch kinematics and the matched
    // reco object's pt/eta/phi, to fill the resolution inputs.
    void fill_resolution(TruthBranchHistograms const& histograms,
                         std::size_t index,
                         Kinematics const& truth,
                         double recoPt,
                         double recoEta,
                         double recoPhi) const;

  private:
    struct Axis {
      int nbins;
      double min, max;
      double linthresh = 0.;
    };
    // Which entries of Kinematics::asVector each side books, in booking order.
    std::vector<std::size_t> truthVars_, recoVars_;
    std::vector<std::string> truthVarNames_, recoVarNames_;
    // Cut bit per truth variable, resolved once so the fill loop does no string work.
    std::vector<uint32_t> truthCutBits_;
    std::vector<Axis> truthAxes_, recoAxes_;

    int nintScore_, nintShared_, nintRes_;
    double minScore_, maxScore_, minShared_, maxShared_, minRes_, maxRes_;
    // The resolution 2D uses its OWN, coarser x binning: each x slice is fitted with a
    // Gaussian, so it needs enough entries per slice to constrain the fit, which the
    // efficiency binning does not provide.
    Axis resEtaAxis_, resPtAxis_;
  };

}  // namespace truth

#endif
