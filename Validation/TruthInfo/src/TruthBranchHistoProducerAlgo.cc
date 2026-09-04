// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include <algorithm>
#include <cmath>
#include <iterator>

#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"
#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"
#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

namespace {
  // One bin per VertexReason, the enum being contiguous from Unknown to Other, plus one
  // synthetic bin. VertexReason is derived from the Geant4 creator-process subtype of a
  // SimVertex, so a GEN-only vertex has no process and reads as Unknown. That is a
  // different statement from "the process is not one we map", and in a pileup sample it
  // is the dominant category: collapsePileupGen replaces each pileup interaction with
  // one GEN-only vertex carrying all its stable particles. Giving it its own bin keeps
  // Unknown meaning what it says.
  constexpr int kNReasons = static_cast<int>(truth::VertexReason::Other) + 1;
  constexpr int kGenOnlyBin = kNReasons;
  constexpr int kNReasonBins = kNReasons + 1;
}  // namespace

namespace truth {

  TruthBranchHistoProducerAlgo::TruthBranchHistoProducerAlgo(edm::ParameterSet const& pset)
      : nintScore_(pset.getParameter<int>("nintScore")),
        nintShared_(pset.getParameter<int>("nintShared")),
        nintRes_(pset.getParameter<int>("nintRes")),
        minScore_(pset.getParameter<double>("minScore")),
        maxScore_(pset.getParameter<double>("maxScore")),
        minShared_(pset.getParameter<double>("minShared")),
        maxShared_(pset.getParameter<double>("maxShared")),
        minRes_(pset.getParameter<double>("minRes")),
        maxRes_(pset.getParameter<double>("maxRes")),
        resEtaAxis_{pset.getParameter<int>("nint_res_eta"),
                    pset.getParameter<double>("min_res_eta"),
                    pset.getParameter<double>("max_res_eta")},
        resPtAxis_{pset.getParameter<int>("nint_res_pt"),
                   pset.getParameter<double>("min_res_pt"),
                   pset.getParameter<double>("max_res_pt")} {
    // Resolve a variable name to its position in Kinematics::asVector, so a typo in the
    // configuration is a configuration error and not a silently missing plot.
    // prefix lets ONE side override a range the other must keep. The truth zpos of a
    // trackster branch is its production vertex, in the tracker; the reco zpos is the
    // trackster barycentre, in HGCal at |z| of 320 to 520 cm. Sharing one range put 100%
    // of reco trackster z in the under and overflow, so that plot drew nothing at all.
    auto resolve = [&](std::vector<std::string> const& names,
                       std::vector<std::size_t>& indices,
                       std::vector<std::string>& kept,
                       std::vector<Axis>& axes,
                       std::string const& prefix = "") {
      for (auto const& name : names) {
        const auto it = std::find(kVariableNames.begin(), kVariableNames.end(), name);
        if (it == kVariableNames.end()) {
          throw cms::Exception("Configuration") << "unknown truth-branch plot variable '" << name << "'";
        }
        indices.push_back(static_cast<std::size_t>(std::distance(kVariableNames.begin(), it)));
        kept.push_back(name);
        const std::string key = (!prefix.empty() && pset.existsAs<int>("nint_" + prefix + name)) ? prefix + name : name;
        axes.push_back({pset.getParameter<int>("nint_" + key),
                        pset.getParameter<double>("min_" + key),
                        pset.getParameter<double>("max_" + key),
                        pset.getParameter<double>("linthresh_" + key)});
      }
    };
    resolve(pset.getParameter<std::vector<std::string>>("truthVariables"), truthVars_, truthVarNames_, truthAxes_);
    resolve(pset.getParameter<std::vector<std::string>>("recoVariables"), recoVars_, recoVarNames_, recoAxes_, "reco_");
    truthCutBits_.reserve(truthVarNames_.size());
    for (auto const& name : truthVarNames_) {
      truthCutBits_.push_back(cutBitOfVariable(name));
    }
  }

  std::vector<float> TruthBranchHistoProducerAlgo::binEdges(SymlogAxis const& axis) {
    std::vector<float> edges;
    edges.reserve(axis.nbins + 1);
    if (axis.linthresh <= 0. || axis.linthresh >= axis.max || axis.nbins < 2) {
      const double width = (axis.max - axis.min) / axis.nbins;
      for (int i = 0; i <= axis.nbins; ++i) {
        edges.push_back(static_cast<float>(axis.min + i * width));
      }
      return edges;
    }
    // One linear bin holding everything below the threshold, zero included, then a log
    // ladder to the top. The first edge stays at min so nothing that used to be in range
    // silently becomes underflow.
    edges.push_back(static_cast<float>(axis.min));
    const int nLog = axis.nbins - 1;
    const double lo = std::log10(axis.linthresh);
    const double hi = std::log10(axis.max);
    for (int i = 0; i <= nLog; ++i) {
      edges.push_back(static_cast<float>(std::pow(10., lo + (hi - lo) * i / nLog)));
    }
    return edges;
  }

  namespace {
    // The ME names are the harvesting API: DQMGenericClient forms every ratio from
    // these by string, so a rename silently drops a plot rather than failing.
    template <typename AxisT>
    void bookRow(dqm::implementation::IBooker& booker,
                 std::vector<TruthBranchHistograms::MERow>& rows,
                 std::string const& prefix,
                 std::vector<std::string> const& names,
                 std::vector<AxisT> const& axes) {
      TruthBranchHistograms::MERow row;
      for (std::size_t v = 0; v < names.size(); ++v) {
        auto const& name = names[v];
        auto const& axis = axes[v];
        const auto edges = TruthBranchHistoProducerAlgo::binEdges({axis.nbins, axis.min, axis.max, axis.linthresh});
        auto* me = booker.book1D(prefix + "_" + name, prefix + " vs " + name, axis.nbins, edges.data());
        // The flavour axis is species, not a number: label it so the DQM GUI reads as
        // d/u/s/c/b/t/g rather than as bin indices.
        if (name == "flavour") {
          for (int f = 0; f < kNFlavourBins && f < axis.nbins; ++f) {
            me->setBinLabel(f + 1, kFlavourBinNames[f]);
          }
        }
        row.push_back(me);
      }
      rows.push_back(std::move(row));
    }
  }  // namespace

  void TruthBranchHistoProducerAlgo::bookTruthHistos(dqm::implementation::IBooker& booker,
                                                     TruthBranchHistograms& h,
                                                     bool calorimetric) const {
    // One block of kNEtaRegions rows per entry, the region ones in sub-folders carrying
    // the SAME ME names, so every harvester string and the plot script work unchanged.
    const std::string base = booker.pwd();
    for (std::size_t r = 0; r < kNEtaRegions; ++r) {
      booker.setCurrentFolder(r == 0 ? base : base + "/" + kEtaRegionFolders[r]);
      bookTruthRow(booker, h, calorimetric);
    }
    booker.setCurrentFolder(base);
    bookTruthDiagnostics(booker, h, calorimetric);
  }

  void TruthBranchHistoProducerAlgo::bookTruthRow(dqm::implementation::IBooker& booker,
                                                  TruthBranchHistograms& h,
                                                  bool calorimetric) const {
    bookRow(booker, h.h_simul, "num_simul", truthVarNames_, truthAxes_);
    bookRow(booker, h.h_assoc_simToReco, "num_assoc(simToReco)", truthVarNames_, truthAxes_);
    bookRow(booker, h.h_assoc_simToReco_cumulative, "num_assoc_cumulative", truthVarNames_, truthAxes_);
    if (!calorimetric) {
      bookRow(booker, h.h_duplicate, "num_duplicate", truthVarNames_, truthAxes_);
    }
    bookRow(booker, h.h_split, "num_split", truthVarNames_, truthAxes_);
  }

  // Booked ONCE per entry, in the base folder, not per region: these are distributions
  // rather than ratio numerators, and their fills index the entry directly. Booking them
  // per region would leave the region copies unfilled and shift every index.
  void TruthBranchHistoProducerAlgo::bookTruthDiagnostics(dqm::implementation::IBooker& booker,
                                                          TruthBranchHistograms& h,
                                                          bool calorimetric) const {
    // Categorical axis: one labelled bin per Geant4 creation process.
    auto bookReason = [&](std::vector<TruthBranchHistograms::METype>& v, std::string const& name) {
      auto* me = booker.book1D(name, name, kNReasonBins, -0.5, kNReasonBins - 0.5);
      for (int r = 0; r < kNReasons; ++r) {
        me->setBinLabel(r + 1, truth::vertexReasonName(static_cast<truth::VertexReason>(r)));
      }
      me->setBinLabel(kGenOnlyBin + 1, "GenOnly");
      v.push_back(me);
    };
    bookReason(h.h_simul_reason, "num_simul_reason");
    bookReason(h.h_assoc_simToReco_reason, "num_assoc(simToReco)_reason");
    if (!calorimetric) {
      bookReason(h.h_duplicate_reason, "num_duplicate_reason");
    }

    // Truth purity: the truth object is the denominator, so it lives on this side.
    h.h_truthPurity.push_back(booker.book1D("truth_purity", "Truth purity", 50, 0., 1.));

    // A fraction of the truth object's own energy, so a [0, 1] axis like truth purity
    // and unlike the reco-side shared quantity, which counts hits or GeV.
    if (calorimetric) {
      h.h_sharedEnergyFraction.push_back(booker.book1D("shared_energy_fraction", "Shared energy fraction", 50, 0., 1.));
    }
  }

  void TruthBranchHistoProducerAlgo::bookRecoHistos(dqm::implementation::IBooker& booker,
                                                    TruthBranchHistograms& h,
                                                    bool calorimetric) const {
    const std::string base = booker.pwd();
    for (std::size_t r = 0; r < kNEtaRegions; ++r) {
      booker.setCurrentFolder(r == 0 ? base : base + "/" + kEtaRegionFolders[r]);
      bookRecoRow(booker, h, calorimetric);
    }
    booker.setCurrentFolder(base);
    bookRecoDiagnostics(booker, h, calorimetric);
  }

  void TruthBranchHistoProducerAlgo::bookRecoRow(dqm::implementation::IBooker& booker,
                                                 TruthBranchHistograms& h,
                                                 bool calorimetric) const {
    bookRow(booker, h.h_reco, "num_reco", recoVarNames_, recoAxes_);
    bookRow(booker, h.h_dominated, "num_dominated", recoVarNames_, recoAxes_);
    bookRow(booker, h.h_levelCandidate, "num_levelcandidate", recoVarNames_, recoAxes_);
    bookRow(booker, h.h_assoc_recoToSim, "num_assoc(recoToSim)", recoVarNames_, recoAxes_);
    bookRow(booker, h.h_recopurity, "num_recopurity", recoVarNames_, recoAxes_);
    bookRow(booker, h.h_pileup, "num_pileup", recoVarNames_, recoAxes_);
    if (calorimetric) {
      bookRow(booker, h.h_assoc_strict, "num_assoc_strict", recoVarNames_, recoAxes_);
    }
  }

  void TruthBranchHistoProducerAlgo::bookRecoDiagnostics(dqm::implementation::IBooker& booker,
                                                         TruthBranchHistograms& h,
                                                         bool calorimetric) const {
    h.h_score.push_back(booker.book1D("association_score", "Association score", nintScore_, minScore_, maxScore_));
    h.h_sharedQuantity.push_back(
        booker.book1D("shared_quantity", "Shared hits or energy", nintShared_, minShared_, maxShared_));
    h.h_leadingShare.push_back(booker.book1D("leading_truth_share", "Leading truth contributor share", 50, 0., 1.));
    h.h_dominanceRatio.push_back(
        booker.book1D("dominance_ratio", "Leading over runner-up truth contributor", 40, 0., 20.));

    // Reco purity: the reco object is the denominator, on a [0, 1] axis.
    h.h_recoPurity.push_back(booker.book1D("reco_purity", "Reco purity", 50, 0., 1.));

    // 2D inputs for the Gaussian slice fit the harvester runs. Same naming as MTV so
    // the resolution strings and the plot script read the same way.
    auto const& etaAxis = resEtaAxis_;
    auto const& ptAxis = resPtAxis_;
    h.h_ptres_vs_eta.push_back(booker.book2D("ptres_vs_eta",
                                             "Relative p_{T} residual vs #eta",
                                             etaAxis.nbins,
                                             etaAxis.min,
                                             etaAxis.max,
                                             nintRes_,
                                             minRes_,
                                             maxRes_));
    h.h_ptres_vs_pt.push_back(booker.book2D("ptres_vs_pt",
                                            "Relative p_{T} residual vs p_{T}",
                                            ptAxis.nbins,
                                            ptAxis.min,
                                            ptAxis.max,
                                            nintRes_,
                                            minRes_,
                                            maxRes_));
    h.h_etares_vs_eta.push_back(booker.book2D(
        "etares_vs_eta", "#eta residual vs #eta", etaAxis.nbins, etaAxis.min, etaAxis.max, nintRes_, minRes_, maxRes_));
    h.h_phires_vs_eta.push_back(booker.book2D(
        "phires_vs_eta", "#phi residual vs #eta", etaAxis.nbins, etaAxis.min, etaAxis.max, nintRes_, minRes_, maxRes_));
  }

  void TruthBranchHistoProducerAlgo::fill_simul(TruthBranchHistograms const& h,
                                                std::size_t i,
                                                Kinematics const& kin,
                                                TruthOutcome outcome,
                                                bool cumulative,
                                                uint32_t failedCuts) const {
    // Inclusive row always, plus the object's region row. The region variable is the one
    // that decides acceptance: where the branch ENTERS the calorimeter when the domain
    // records that, since a branch produced centrally can deposit in an endcap.
    const double regionEta = (kin.caloeta != kNoCaloEntry) ? kin.caloeta : kin.eta;
    const auto region = etaRegionOf(std::abs(regionEta));
    fill_simul_row(h, kNEtaRegions * i, kin, outcome, cumulative, failedCuts);
    if (region != EtaRegion::Inclusive) {
      fill_simul_row(h, kNEtaRegions * i + static_cast<std::size_t>(region), kin, outcome, cumulative, failedCuts);
    }
  }

  uint32_t TruthBranchHistoProducerAlgo::cutBitOfVariable(std::string const& name) {
    if (name == "pt")
      return static_cast<uint32_t>(BranchSelector::CutBit::Pt);
    if (name == "eta")
      return static_cast<uint32_t>(BranchSelector::CutBit::Eta);
    return 0u;
  }

  void TruthBranchHistoProducerAlgo::fill_simul_row(TruthBranchHistograms const& h,
                                                    std::size_t i,
                                                    Kinematics const& kin,
                                                    TruthOutcome outcome,
                                                    bool cumulative,
                                                    uint32_t failedCuts) const {
    const auto values = kin.asVector();
    for (std::size_t v = 0; v < truthVars_.size(); ++v) {
      // Variable-blind: this object may enter the plot only if the cuts it fails are
      // exactly the cut on THIS variable, so the axis never has its own cut applied and
      // no other axis is polluted by an object a cut would have removed.
      if ((failedCuts & ~truthCutBits_[v]) != 0u) {
        continue;
      }
      const double x = values[truthVars_[v]];
      h.h_simul[i][v]->Fill(x);
      // Individual and Duplicate both mean the truth object WAS reconstructed as one
      // object, so both count in the efficiency numerator; they differ in whether it
      // happened once or more than once. Split did not happen as one object at all.
      if (outcome == TruthOutcome::Individual || outcome == TruthOutcome::Duplicate) {
        h.h_assoc_simToReco[i][v]->Fill(x);
      }
      if (cumulative) {
        h.h_assoc_simToReco_cumulative[i][v]->Fill(x);
      }
      if (outcome == TruthOutcome::Duplicate && !h.h_duplicate.empty()) {
        h.h_duplicate[i][v]->Fill(x);
      }
      if (outcome == TruthOutcome::Split) {
        h.h_split[i][v]->Fill(x);
      }
    }
  }

  void TruthBranchHistoProducerAlgo::fill_truth_purity(TruthBranchHistograms const& h,
                                                       std::size_t i,
                                                       double truthPurity) const {
    h.h_truthPurity[i]->Fill(truthPurity);
  }

  void TruthBranchHistoProducerAlgo::fill_shared_energy_fraction(TruthBranchHistograms const& h,
                                                                 std::size_t i,
                                                                 double sharedEnergyFraction) const {
    h.h_sharedEnergyFraction[i]->Fill(sharedEnergyFraction);
  }

  void TruthBranchHistoProducerAlgo::fill_reco(TruthBranchHistograms const& h,
                                               std::size_t i,
                                               Kinematics const& kin,
                                               RecoOutcome const& outcome) const {
    const auto region = etaRegionOf(std::abs(kin.eta));
    fill_reco_row(h, kNEtaRegions * i, kin, outcome);
    if (region != EtaRegion::Inclusive) {
      fill_reco_row(h, kNEtaRegions * i + static_cast<std::size_t>(region), kin, outcome);
    }
  }

  void TruthBranchHistoProducerAlgo::fill_reco_row(TruthBranchHistograms const& h,
                                                   std::size_t i,
                                                   Kinematics const& kin,
                                                   RecoOutcome const& outcome) const {
    const auto values = kin.asVector();
    for (std::size_t v = 0; v < recoVars_.size(); ++v) {
      const double x = values[recoVars_[v]];
      h.h_reco[i][v]->Fill(x);
      if (outcome.dominated) {
        h.h_dominated[i][v]->Fill(x);
      }
      if (outcome.hasLevelCandidate) {
        h.h_levelCandidate[i][v]->Fill(x);
      }
      if (outcome.associated) {
        h.h_assoc_recoToSim[i][v]->Fill(x);
        h.h_recopurity[i][v]->Fill(x, outcome.matchQuality);
      }
      if (outcome.pileup) {
        h.h_pileup[i][v]->Fill(x);
      }
      if (outcome.strictMatch && !h.h_assoc_strict.empty()) {
        h.h_assoc_strict[i][v]->Fill(x);
      }
    }
  }

  void TruthBranchHistoProducerAlgo::fill_reason(TruthBranchHistograms const& h,
                                                 std::size_t i,
                                                 unsigned int reason,
                                                 TruthOutcome outcome) const {
    const bool associated = (outcome == TruthOutcome::Individual || outcome == TruthOutcome::Duplicate);
    const bool duplicate = (outcome == TruthOutcome::Duplicate);
    const double bin =
        (reason < static_cast<unsigned int>(kNReasonBins)) ? reason : static_cast<double>(truth::VertexReason::Other);
    h.h_simul_reason[i]->Fill(bin);
    if (associated) {
      h.h_assoc_simToReco_reason[i]->Fill(bin);
    }
    if (duplicate && !h.h_duplicate_reason.empty()) {
      h.h_duplicate_reason[i]->Fill(bin);
    }
  }

  void TruthBranchHistoProducerAlgo::fill_dominance(TruthBranchHistograms const& h,
                                                    std::size_t i,
                                                    double leadingShare,
                                                    double dominanceRatio) const {
    if (leadingShare < 0.) {
      return;
    }
    h.h_leadingShare[i]->Fill(leadingShare);
    h.h_dominanceRatio[i]->Fill(dominanceRatio);
  }

  void TruthBranchHistoProducerAlgo::fill_match(
      TruthBranchHistograms const& h, std::size_t i, double score, double sharedQuantity, double recoPurity) const {
    h.h_score[i]->Fill(score);
    h.h_sharedQuantity[i]->Fill(sharedQuantity);
    h.h_recoPurity[i]->Fill(recoPurity);
  }

  void TruthBranchHistoProducerAlgo::fill_resolution(TruthBranchHistograms const& h,
                                                     std::size_t i,
                                                     Kinematics const& truthKin,
                                                     double recoPt,
                                                     double recoEta,
                                                     double recoPhi) const {
    if (truthKin.pt > 0.) {
      const double dpt = (recoPt - truthKin.pt) / truthKin.pt;
      h.h_ptres_vs_eta[i]->Fill(truthKin.eta, dpt);
      h.h_ptres_vs_pt[i]->Fill(truthKin.pt, dpt);
    }
    h.h_etares_vs_eta[i]->Fill(truthKin.eta, recoEta - truthKin.eta);
    h.h_phires_vs_eta[i]->Fill(truthKin.eta, recoPhi - truthKin.phi);
  }

}  // namespace truth
