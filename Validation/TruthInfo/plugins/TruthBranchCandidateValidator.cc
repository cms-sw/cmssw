// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// TICL candidate validation against the truth graph. A candidate is two channels: its calorimeter part, matched by
// truthBranchCandidateAssociators on the layer clusters of its tracksters, and its track, matched by the track
// associator. The calorimeter identity of a candidate is the branch an adaptive working point chose for its
// calorimeter part; the track identity is the particle of its track.
//
// Truth-driven folders, one per (collection, working point, class, level): a level member is reconstructed when the
// calorimeter or the track identity of at least one candidate projects on it, at or below it in the graph. The
// ladder then asks, on the representative candidate, for the right track, the right class and an energy response
// in a window. A member that no candidate projects on, but that is below an identity of a candidate, is merged.
// num_multiple counts a member with two or more candidates on it, which also counts as reconstructed.
//
// These definitions differ from the trackster ones of TruthBranchRecoValidator: there is no shared-energy cut, and a
// multiple member is also a reconstructed member, where a trackster split member is not reconstructed.
//
// Reco-driven folders, one per (collection, class, working point): fake, no-candidate, pileup and purity as in
// TruthBranchRecoValidator, on the calorimeter identity, or on the track identity for a candidate without a
// trackster, and the track and class checks of each candidate against that identity.
//
// Every ratio is formed by DQMGenericClient from the num_* names.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"
#include "PhysicsTools/TruthInfo/interface/SubgraphHitView.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "Validation/TruthInfo/interface/CandidateTruthMatching.h"
#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

namespace {
  using truth::candidate::kNClasses;
  using truth::candidate::kNIdentityBins;
  using truth::candidate::LevelProjection;
  using truth::candidate::ParticleClass;
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  using ME = dqm::reco::MonitorElement*;
  constexpr uint32_t kNone = truth::candidate::kNoParticle;
  // The ncandidates histogram has one bin per count from 0; the last bin also holds the larger counts.
  constexpr int kNCandidateBins = 10;

  enum Variable { kPt, kEta, kPhi, kEnergy, kNVariables };
  constexpr std::array<const char*, kNVariables> kVariableNames{"pt", "eta", "phi", "energy"};

  // The ME names are the harvesting API.
  enum TruthCount { kSimul, kAssoc, kAssocTrack, kAssocPid, kAssocEnergy, kMultiple, kMerged, kNTruthCounts };
  constexpr std::array<const char*, kNTruthCounts> kTruthCountNames{"num_simul",
                                                                    "num_assoc(simToReco)",
                                                                    "num_assoc_track",
                                                                    "num_assoc_pid",
                                                                    "num_assoc_energy",
                                                                    "num_multiple",
                                                                    "num_merged"};
  enum RecoCount {
    kReco,
    kAssocReco,
    kDominated,
    kLevelCandidate,
    kPileup,
    kRecoPurity,
    kCharged,
    kTrackMatched,
    kTrackCalo,
    kTrackConsistent,
    kLevelAssoc,
    kPidCorrect,
    kMergedCandidate,
    kNRecoCounts
  };
  constexpr std::array<const char*, kNRecoCounts> kRecoCountNames{"num_reco",
                                                                  "num_assoc(recoToSim)",
                                                                  "num_dominated",
                                                                  "num_levelcandidate",
                                                                  "num_pileup",
                                                                  "num_recopurity",
                                                                  "num_charged",
                                                                  "num_trackmatched",
                                                                  "num_trackcalo",
                                                                  "num_trackconsistent",
                                                                  "num_levelassoc",
                                                                  "num_pidcorrect",
                                                                  "num_mergedcandidate"};
  // Residuals of the representative candidate, (reco - truth) / truth for the energies and the pt, reco - truth
  // for the angles, against a truth variable.
  enum Residual {
    kEresVsEnergy,
    kEresVsEta,
    kRawEresVsEnergy,
    kEsumresVsEnergy,
    kPtresVsPt,
    kEtaresVsEta,
    kPhiresVsEta,
    kNResiduals
  };
  constexpr std::array<const char*, kNResiduals> kResidualNames{"eres_vs_energy",
                                                                "eres_vs_eta",
                                                                "rawres_vs_energy",
                                                                "esumres_vs_energy",
                                                                "ptres_vs_pt",
                                                                "etares_vs_eta",
                                                                "phires_vs_eta"};
  // nCandidates, score and leadingShare are booked in the inclusive folder only, confusion in the folders of all
  // classes; else they are nullptr.
  struct TruthFolder {
    std::array<std::array<ME, kNVariables>, kNTruthCounts> counts{};
    std::array<ME, kNResiduals> residuals{};
    ME nCandidates = nullptr;
    ME confusion = nullptr;
  };
  struct RecoFolder {
    std::array<std::array<ME, kNVariables>, kNRecoCounts> counts{};
    ME score = nullptr;
    ME leadingShare = nullptr;
    ME confusion = nullptr;
  };

  struct TruthBranchCandidateHistograms {
    std::vector<TruthFolder> truth;
    std::vector<RecoFolder> reco;
  };

  // Label and instance joined by an underscore, the key of the association products and of the folders.
  std::string instanceKey(edm::InputTag const& tag) {
    return tag.instance().empty() ? tag.label() : tag.label() + "_" + tag.instance();
  }
}  // namespace

class TruthBranchCandidateValidator : public DQMGlobalEDAnalyzer<TruthBranchCandidateHistograms> {
public:
  using Histograms = TruthBranchCandidateHistograms;

  explicit TruthBranchCandidateValidator(edm::ParameterSet const&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

  struct Axis {
    int nbins = 1;
    double min = 0.;
    double max = 1.;
    double linthresh = 0.;
  };
  struct Collection {
    std::string key;
    edm::EDGetTokenT<std::vector<TICLCandidate>> candidates;
    edm::EDGetTokenT<MapType> reference;
    edm::EDGetTokenT<MapType> truthToReco;
    std::vector<edm::EDGetTokenT<MapType>> perWorkingPoint;
  };
  struct LevelTokens {
    std::string name;
    edm::EDGetTokenT<std::vector<unsigned int>> targets;
    edm::EDGetTokenT<std::vector<unsigned int>> eligibility;
  };

  [[nodiscard]] std::size_t truthIndex(std::size_t k, std::size_t w, std::size_t l, std::size_t c, std::size_t r) const {
    return (((k * workingPoints_.size() + w) * levels_.size() + l) * (kNClasses + 1) + c) * nRegionSlots() + r;
  }
  [[nodiscard]] std::size_t recoIndex(std::size_t k, std::size_t w, std::size_t c, std::size_t r) const {
    return ((k * workingPoints_.size() + w) * (kNClasses + 1) + c) * nRegionSlots() + r;
  }
  [[nodiscard]] std::size_t nRegionSlots() const { return 1 + regions_.size(); }
  // The booked folder slot of an |eta|, 0 when its region is not booked.
  [[nodiscard]] std::size_t regionSlot(double absEta) const {
    return regionSlot_[static_cast<std::size_t>(truth::etaRegionOf(absEta))];
  }
  // The folder slots an object fills: the inclusive one, and the one of its region when that is booked.
  struct Slots {
    std::array<std::size_t, 2> slot{};
    std::size_t n = 1;
    [[nodiscard]] auto begin() const { return slot.begin(); }
    [[nodiscard]] auto end() const { return slot.begin() + n; }
  };
  [[nodiscard]] Slots slotsOf(double absEta) const {
    Slots slots;
    if (const std::size_t r = regionSlot(absEta); r != 0)
      slots.slot[slots.n++] = r;
    return slots;
  }
  void bookCounts(DQMStore::IBooker&, std::string const& name, std::array<ME, kNVariables>&) const;

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracksToken_;
  const edm::EDGetTokenT<MapType> trackMapToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> assignableToken_;
  const std::string dirName_;
  const std::vector<std::string> workingPoints_;
  std::vector<LevelTokens> levels_;
  std::size_t dominanceLevel_ = 0;
  std::vector<Collection> collections_;
  const double minLeadingTruthShare_;
  const double minTrackPurity_;
  // A level member must have a calorimeter subgraph hit in requireHitsDetectors_.
  const bool requireHits_;
  const uint32_t requireHitsDetectors_;
  const truth::candidate::LadderConfig ladder_;
  std::array<Axis, kNVariables> axes_;
  std::array<uint32_t, kNVariables> cutBits_{};
  Axis resAxis_, resEnergyAxis_, resEtaAxis_, resPtAxis_;
  std::vector<truth::EtaRegion> regions_;
  std::array<std::size_t, truth::kNEtaRegions> regionSlot_{};
  // A missing input leaves its folders booked and empty for the whole job, so it is reported once.
  mutable std::once_flag levelsMissingWarned_;
  mutable std::once_flag tracksMissingWarned_;
  mutable std::once_flag foreignTrackWarned_;
  mutable std::vector<std::once_flag> collectionMissingWarned_;
};

TruthBranchCandidateValidator::TruthBranchCandidateValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      tracksToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      trackMapToken_(
          consumes<MapType>(edm::InputTag(cfg.getParameter<std::string>("trackAssociator"),
                                          instanceKey(cfg.getParameter<edm::InputTag>("tracks")) + "RecoToTruth" +
                                              cfg.getParameter<std::string>("referenceWorkingPoint")))),
      assignableToken_(consumes<std::vector<unsigned int>>(
          edm::InputTag(cfg.getParameter<std::string>("targetsProducer"), "assignableRoots"))),
      dirName_(cfg.getParameter<std::string>("dirName")),
      workingPoints_(cfg.getParameter<std::vector<std::string>>("workingPoints")),
      minLeadingTruthShare_(cfg.getParameter<double>("minLeadingTruthShare")),
      minTrackPurity_(cfg.getParameter<double>("minTrackPurity")),
      requireHits_(cfg.getParameter<bool>("requireHits")),
      requireHitsDetectors_(
          truth::BranchHitAssociator::detectorMask(cfg.getParameter<std::vector<std::string>>("denominatorDetectors"))),
      ladder_{cfg.getParameter<double>("energyResponseMin"), cfg.getParameter<double>("energyResponseMax")} {
  if (workingPoints_.empty())
    throw cms::Exception("Configuration") << "workingPoints is empty";
  const auto targetsProducer = cfg.getParameter<std::string>("targetsProducer");
  const auto levelNames = cfg.getParameter<std::vector<std::string>>("truthLevels");
  const auto dominance = cfg.getParameter<std::string>("dominanceLevel");
  for (auto const& name : levelNames) {
    if (name.empty())
      throw cms::Exception("Configuration") << "truthLevels has an empty name";
    std::string capitalized = name;
    capitalized[0] = std::toupper(static_cast<unsigned char>(capitalized[0]));
    const std::string instance = "truthToRecoTargets" + capitalized;
    levels_.push_back({name,
                       consumes<std::vector<unsigned int>>(edm::InputTag(targetsProducer, instance)),
                       consumes<std::vector<unsigned int>>(edm::InputTag(targetsProducer, instance + "Eligibility"))});
  }
  const auto dominanceIt = std::find(levelNames.begin(), levelNames.end(), dominance);
  if (dominanceIt == levelNames.end())
    throw cms::Exception("Configuration") << "dominanceLevel '" << dominance << "' is not in truthLevels";
  dominanceLevel_ = static_cast<std::size_t>(dominanceIt - levelNames.begin());

  const auto associator = cfg.getParameter<std::string>("associator");
  const auto reference = cfg.getParameter<std::string>("referenceWorkingPoint");
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recoCollections")) {
    Collection collection;
    collection.key = instanceKey(tag);
    collection.candidates = consumes<std::vector<TICLCandidate>>(tag);
    collection.reference = consumes<MapType>(edm::InputTag(associator, collection.key + "RecoToTruth" + reference));
    collection.truthToReco = consumes<MapType>(edm::InputTag(associator, collection.key + "TruthToReco"));
    for (auto const& wp : workingPoints_)
      collection.perWorkingPoint.push_back(
          consumes<MapType>(edm::InputTag(associator, collection.key + "RecoToTruth" + wp)));
    collections_.push_back(std::move(collection));
  }
  collectionMissingWarned_ = std::vector<std::once_flag>(collections_.size());

  auto const& block = cfg.getParameter<edm::ParameterSet>("histoBlock");
  for (int v = 0; v < kNVariables; ++v) {
    const std::string name = kVariableNames[v];
    axes_[v] = {block.getParameter<int>("nint_" + name),
                block.getParameter<double>("min_" + name),
                block.getParameter<double>("max_" + name),
                block.getParameter<double>("linthresh_" + name)};
    cutBits_[v] = truth::TruthBranchHistoProducerAlgo::cutBitOfVariable(name);
  }
  auto residualAxis = [&block](std::string const& name) {
    return Axis{block.getParameter<int>("nint_" + name),
                block.getParameter<double>("min_" + name),
                block.getParameter<double>("max_" + name),
                0.};
  };
  resAxis_ = residualAxis("res");
  resEnergyAxis_ = residualAxis("res_energy");
  resEtaAxis_ = residualAxis("res_eta");
  resPtAxis_ = residualAxis("res_pt");
  for (auto const& name : block.getParameter<std::vector<std::string>>("etaRegions")) {
    const auto it = std::find(truth::kEtaRegionFolders.begin() + 1, truth::kEtaRegionFolders.end(), name);
    if (it == truth::kEtaRegionFolders.end())
      throw cms::Exception("Configuration") << "unknown eta region '" << name << "'";
    const auto region = static_cast<truth::EtaRegion>(it - truth::kEtaRegionFolders.begin());
    regions_.push_back(region);
    regionSlot_[static_cast<std::size_t>(region)] = regions_.size();
  }
}

void TruthBranchCandidateValidator::bookCounts(DQMStore::IBooker& booker,
                                               std::string const& name,
                                               std::array<ME, kNVariables>& row) const {
  for (int v = 0; v < kNVariables; ++v) {
    auto const& a = axes_[v];
    const auto edges = truth::TruthBranchHistoProducerAlgo::binEdges({a.nbins, a.min, a.max, a.linthresh});
    row[v] = booker.book1D(name + "_" + kVariableNames[v], name + " vs " + kVariableNames[v], a.nbins, edges.data());
  }
}

void TruthBranchCandidateValidator::bookHistograms(DQMStore::IBooker& booker,
                                                   edm::Run const&,
                                                   edm::EventSetup const&,
                                                   Histograms& histograms) const {
  auto regionSuffix = [this](std::size_t r) {
    return r == 0 ? std::string() : "/" + truth::kEtaRegionFolders[static_cast<std::size_t>(regions_[r - 1])];
  };
  auto classInfix = [](std::size_t c) {
    return c == 0 ? std::string() : std::string("_") + truth::candidate::kClassNames[c - 1];
  };
  auto labelClasses = [](ME me, int axis) {
    for (std::size_t c = 0; c < kNClasses; ++c)
      me->setBinLabel(c + 1, truth::candidate::kClassNames[c], axis);
  };

  histograms.truth.resize(collections_.size() * workingPoints_.size() * levels_.size() * (kNClasses + 1) *
                          nRegionSlots());
  histograms.reco.resize(collections_.size() * workingPoints_.size() * (kNClasses + 1) * nRegionSlots());
  for (std::size_t k = 0; k < collections_.size(); ++k) {
    for (std::size_t w = 0; w < workingPoints_.size(); ++w) {
      for (std::size_t c = 0; c <= kNClasses; ++c) {
        for (std::size_t r = 0; r < nRegionSlots(); ++r) {
          booker.setCurrentFolder(dirName_ + collections_[k].key + classInfix(c) + "_" + workingPoints_[w] +
                                  regionSuffix(r));
          auto& f = histograms.reco[recoIndex(k, w, c, r)];
          for (int n = 0; n < kNRecoCounts; ++n)
            bookCounts(booker, kRecoCountNames[n], f.counts[n]);
          if (c != 0)
            continue;
          if (r == 0) {
            f.score = booker.book1D("score", "Score of the calorimeter match;score;candidates", 50, 0., 1.);
            f.leadingShare = booker.book1D(
                "leading_truth_share", "Share of the leading level member;share;candidates", 51, 0., 1.02);
          }
          f.confusion = booker.book2D("pid_confusion",
                                      "Candidate class vs identity class;candidate;identity",
                                      kNClasses,
                                      0.,
                                      kNClasses,
                                      kNIdentityBins,
                                      0.,
                                      kNIdentityBins);
          labelClasses(f.confusion, 1);
          labelClasses(f.confusion, 2);
          f.confusion->setBinLabel(truth::candidate::kIdentityMerged + 1, "merged", 2);
          f.confusion->setBinLabel(truth::candidate::kIdentityPileup + 1, "pileup", 2);
          f.confusion->setBinLabel(truth::candidate::kIdentityOutsideLevel + 1, "outside level", 2);
          f.confusion->setBinLabel(truth::candidate::kIdentityNone + 1, "unmatched", 2);
        }
      }
      for (std::size_t l = 0; l < levels_.size(); ++l) {
        for (std::size_t c = 0; c <= kNClasses; ++c) {
          for (std::size_t r = 0; r < nRegionSlots(); ++r) {
            booker.setCurrentFolder(dirName_ + collections_[k].key + "_" + workingPoints_[w] + classInfix(c) + "_" +
                                    levels_[l].name + regionSuffix(r));
            auto& f = histograms.truth[truthIndex(k, w, l, c, r)];
            for (int n = 0; n < kNTruthCounts; ++n)
              bookCounts(booker, kTruthCountNames[n], f.counts[n]);
            const std::array<Axis const*, kNResiduals> x{&resEnergyAxis_,
                                                         &resEtaAxis_,
                                                         &resEnergyAxis_,
                                                         &resEnergyAxis_,
                                                         &resPtAxis_,
                                                         &resEtaAxis_,
                                                         &resEtaAxis_};
            for (int n = 0; n < kNResiduals; ++n)
              f.residuals[n] = booker.book2D(kResidualNames[n],
                                             kResidualNames[n],
                                             x[n]->nbins,
                                             x[n]->min,
                                             x[n]->max,
                                             resAxis_.nbins,
                                             resAxis_.min,
                                             resAxis_.max);
            if (c != 0)
              continue;
            if (r == 0)
              f.nCandidates = booker.book1D("ncandidates",
                                            "Candidates on one member;candidates;members",
                                            kNCandidateBins,
                                            -0.5,
                                            kNCandidateBins - 0.5);
            f.confusion = booker.book2D("pid_confusion",
                                        "Member class vs candidate class;member;candidate",
                                        kNClasses,
                                        0.,
                                        kNClasses,
                                        kNClasses,
                                        0.,
                                        kNClasses);
            labelClasses(f.confusion, 1);
            labelClasses(f.confusion, 2);
          }
        }
      }
    }
  }
}

void TruthBranchCandidateValidator::dqmAnalyze(edm::Event const& event,
                                               edm::EventSetup const&,
                                               Histograms const& histograms) const {
  auto const& graph = event.get(graphToken_);
  const uint32_t nParticles = graph.nParticles();
  truth::SubgraphHitView hitIndex(event.get(hitIndexToken_));
  // A member fills the eta-band folder of the eta where its branch enters the calorimeter, else of its own eta.
  const std::vector<double> caloEntryEta = truth::caloEntryEtas(graph);

  // The members of every level, their eligibility and the projection of the graph on them.
  struct Level {
    std::vector<unsigned int> const* members = nullptr;
    std::vector<unsigned int> const* eligibility = nullptr;
    std::vector<ParticleClass> memberClass;
    LevelProjection projection;
  };
  std::vector<Level> levels;
  levels.reserve(levels_.size());
  for (auto const& tokens : levels_) {
    edm::Handle<std::vector<unsigned int>> targets, eligibility;
    event.getByToken(tokens.targets, targets);
    event.getByToken(tokens.eligibility, eligibility);
    if (!targets.isValid()) {
      std::call_once(levelsMissingWarned_, [&tokens] {
        edm::LogWarning("TruthBranchCandidateValidator")
            << "no targets for level " << tokens.name << "; its folders stay empty for the whole job.";
      });
      return;
    }
    Level level{targets.product(),
                eligibility.isValid() && eligibility->size() == targets->size() ? eligibility.product() : nullptr,
                {},
                LevelProjection(graph, *targets)};
    level.memberClass.reserve(targets->size());
    for (const unsigned int m : *targets)
      level.memberClass.push_back(
          m < nParticles ? truth::candidate::truthClass(graph.particles()[m].pdgId, graph.particle(m).charge())
                         : ParticleClass::Other);
    levels.push_back(std::move(level));
  }
  auto const& dominance = levels[dominanceLevel_];
  const truth::candidate::DominanceConfig dominanceConfig{dominance.memberClass, minLeadingTruthShare_};

  edm::Handle<std::vector<reco::Track>> tracks;
  event.getByToken(tracksToken_, tracks);
  edm::Handle<MapType> trackMap;
  event.getByToken(trackMapToken_, trackMap);
  edm::Handle<std::vector<unsigned int>> assignable;
  event.getByToken(assignableToken_, assignable);
  const bool trackInputs = tracks.isValid() && trackMap.isValid() && assignable.isValid();
  if (!trackInputs) {
    std::call_once(tracksMissingWarned_, [] {
      edm::LogWarning("TruthBranchCandidateValidator")
          << "no tracks, no track association map or no assignable roots; every candidate track reads as unmatched "
             "for the whole job.";
    });
  }
  // The roots a track may be assigned to. Row [0] of the reference track map is a barred root, such as a parton or a
  // beam particle, when no assignable root matches the track.
  std::vector<uint8_t> isAssignable;
  if (trackInputs) {
    isAssignable.assign(nParticles, 0);
    for (const unsigned int id : *assignable)
      if (id < nParticles)
        isAssignable[id] = 1;
  }

  for (std::size_t k = 0; k < collections_.size(); ++k) {
    auto const& col = collections_[k];
    edm::Handle<std::vector<TICLCandidate>> candidatesHandle;
    event.getByToken(col.candidates, candidatesHandle);
    edm::Handle<MapType> referenceHandle, truthToRecoHandle;
    event.getByToken(col.reference, referenceHandle);
    event.getByToken(col.truthToReco, truthToRecoHandle);
    std::vector<MapType const*> perWp;
    for (auto const& token : col.perWorkingPoint) {
      edm::Handle<MapType> h;
      event.getByToken(token, h);
      perWp.push_back(h.isValid() ? h.product() : nullptr);
    }
    if (!candidatesHandle.isValid() || !referenceHandle.isValid() || !truthToRecoHandle.isValid() ||
        std::find(perWp.begin(), perWp.end(), nullptr) != perWp.end()) {
      std::call_once(collectionMissingWarned_[k], [&col] {
        edm::LogWarning("TruthBranchCandidateValidator")
            << "no candidates or no association map for " << col.key << "; its folders stay empty for the whole job.";
      });
      continue;
    }
    auto const& candidates = *candidatesHandle;
    auto const& reference = referenceHandle->getMap();
    auto const& truthToReco = truthToRecoHandle->getMap();
    const std::size_t nCandidates = candidates.size();

    // Working-point-free properties of each candidate.
    struct Info {
      ParticleClass cls = ParticleClass::Other;
      std::array<double, kNVariables> x{};
      bool hasTrack = false;
      bool hasTrackster = false;
      uint32_t trackParticle = kNone;
      float trackPurity = 0.f;
      double leadingShare = -1.;
    };
    std::vector<Info> info(nCandidates);
    std::vector<truth::candidate::RowMatch> row;
    for (std::size_t c = 0; c < nCandidates; ++c) {
      auto const& cand = candidates[c];
      auto& in = info[c];
      in.cls = truth::candidate::candidateClass(cand.pdgId());
      in.x = {cand.pt(), cand.eta(), cand.phi(), cand.energy()};
      in.hasTrackster = !cand.tracksters().empty();
      auto const track = cand.trackPtr();
      in.hasTrack = track.isNonnull();
      if (in.hasTrack && trackInputs) {
        if (track.id() != tracks.id()) {
          std::call_once(foreignTrackWarned_, [] {
            edm::LogWarning("TruthBranchCandidateValidator")
                << "a candidate track is not in the configured track collection; such tracks read as unmatched.";
          });
        } else if (track.key() < trackMap->getMap().size() && !trackMap->getMap()[track.key()].empty()) {
          // [0] is the best particle of the track, the descendant first on equal scores.
          auto const& best = trackMap->getMap()[track.key()][0];
          in.trackPurity = 1.f - best.score();
          if (in.trackPurity >= minTrackPurity_ && best.index() < nParticles && isAssignable[best.index()] != 0)
            in.trackParticle = best.index();
        }
      }
      // Dominance is read on the reference map, which keeps every candidate branch.
      row.clear();
      if (c < reference.size())
        for (auto const& match : reference[c])
          row.push_back({match.index(), match.sharedEnergy()});
      in.leadingShare = truth::candidate::leadingShare(dominance.projection, row, in.hasTrackster, in.trackParticle);
    }

    std::vector<uint32_t> caloIdentity(nCandidates);
    std::vector<truth::candidate::MemberCandidate> onMember;
    for (std::size_t w = 0; w < workingPoints_.size(); ++w) {
      auto const& map = perWp[w]->getMap();
      for (std::size_t c = 0; c < nCandidates; ++c) {
        const bool calo = c < map.size() && !map[c].empty() && map[c][0].index() < nParticles;
        caloIdentity[c] = calo ? map[c][0].index() : kNone;
      }

      // Reco-driven: every candidate, in its class folder and in the inclusive one.
      for (std::size_t c = 0; c < nCandidates; ++c) {
        auto const& in = info[c];
        const auto out =
            truth::candidate::classifyCandidate(graph,
                                                dominance.projection,
                                                dominanceConfig,
                                                {.candidateClass = in.cls,
                                                 .hasTrackster = in.hasTrackster,
                                                 .caloIdentity = caloIdentity[c],
                                                 .caloScore = caloIdentity[c] != kNone ? map[c][0].score() : 1.f,
                                                 .hasTrack = in.hasTrack,
                                                 .trackParticle = in.trackParticle,
                                                 .trackPurity = in.trackPurity,
                                                 .leadingShare = in.leadingShare});

        const std::array<std::size_t, 2> classes{0, static_cast<std::size_t>(in.cls) + 1};
        for (const std::size_t cl : classes) {
          for (const std::size_t r : slotsOf(std::abs(in.x[kEta]))) {
            auto const& f = histograms.reco[recoIndex(k, w, cl, r)];
            auto fill = [&](RecoCount n, double weight = 1.) {
              for (int v = 0; v < kNVariables; ++v)
                f.counts[n][v]->Fill(in.x[v], weight);
            };
            fill(kReco);
            if (out.associated)
              fill(kAssocReco);
            if (out.dominated)
              fill(kDominated);
            if (out.hasLevelCandidate)
              fill(kLevelCandidate);
            if (out.pileup)
              fill(kPileup);
            fill(kRecoPurity, out.purity);
            if (in.hasTrack)
              fill(kCharged);
            if (in.trackParticle != kNone)
              fill(kTrackMatched);
            if (out.trackCalo)
              fill(kTrackCalo);
            if (out.trackConsistent)
              fill(kTrackConsistent);
            if (out.levelAssoc)
              fill(kLevelAssoc);
            if (out.pidCorrect)
              fill(kPidCorrect);
            if (out.merged)
              fill(kMergedCandidate);
            if (f.score != nullptr && out.calo)
              f.score->Fill(map[c][0].score());
            if (f.leadingShare != nullptr && out.hasLevelCandidate)
              f.leadingShare->Fill(in.leadingShare);
            if (f.confusion != nullptr)
              f.confusion->Fill(static_cast<double>(in.cls) + 0.5, static_cast<double>(out.identityBin) + 0.5);
          }
        }
      }

      // Truth-driven: the members of every level.
      for (std::size_t l = 0; l < levels.size(); ++l) {
        auto const& level = levels[l];
        auto const& members = *level.members;
        std::vector<std::vector<uint32_t>> belonging(members.size());
        std::vector<uint32_t> mergedIdentities;
        for (std::size_t c = 0; c < nCandidates; ++c) {
          const uint32_t caloSlot = level.projection.memberSlot(caloIdentity[c]);
          const uint32_t trackSlot = level.projection.memberSlot(info[c].trackParticle);
          if (caloSlot != LevelProjection::kNoMember)
            belonging[caloSlot].push_back(c);
          // A candidate whose two channels are on one member is on it once.
          if (trackSlot != LevelProjection::kNoMember && trackSlot != caloSlot)
            belonging[trackSlot].push_back(c);
          for (const uint32_t id : {caloIdentity[c], info[c].trackParticle})
            if (id != kNone && level.projection.memberSlot(id) == LevelProjection::kNoMember &&
                level.projection.isAbove(id))
              mergedIdentities.push_back(id);
        }
        std::sort(mergedIdentities.begin(), mergedIdentities.end());
        mergedIdentities.erase(std::unique(mergedIdentities.begin(), mergedIdentities.end()), mergedIdentities.end());

        for (std::size_t t = 0; t < members.size(); ++t) {
          const uint32_t p = members[t];
          if (p >= nParticles || truth::isInvisible(graph.particles()[p].pdgId))
            continue;
          if (requireHits_ && !truth::BranchHitAssociator::hasHitInDetectors(
                                  hitIndex.subgraphHits(truth::HitChannel::Calo, p), requireHitsDetectors_))
            continue;
          auto const& p4 = graph.particles()[p].momentum;
          const bool hasDirection = p4.pt() > 0.;
          const std::array<double, kNVariables> x{p4.pt(),
                                                  hasDirection ? p4.eta() : truth::kNoCaloEntry,
                                                  hasDirection ? p4.phi() : truth::kNoCaloEntry,
                                                  p4.energy()};
          const uint32_t failed = level.eligibility != nullptr ? (*level.eligibility)[t] : 0u;
          const ParticleClass memberClass = level.memberClass[t];

          onMember.clear();
          for (const uint32_t c : belonging[t]) {
            truth::candidate::MemberCandidate mc;
            if (p < truthToReco.size()) {
              for (auto const& e : truthToReco[p]) {
                if (e.index() == c) {
                  mc.sharedEnergyFraction = e.sharedEnergy();
                  break;
                }
              }
            }
            mc.candidateClass = info[c].cls;
            mc.hasTrack = info[c].hasTrack;
            mc.trackOnMember =
                info[c].trackParticle != kNone && level.projection.memberSlot(info[c].trackParticle) == t;
            mc.energy = candidates[c].energy();
            onMember.push_back(mc);
          }
          const auto outcome = truth::candidate::evaluateMember(memberClass, p4.energy(), onMember, ladder_);
          const bool merged = !outcome.reconstructed && truth::candidate::hasAncestorIn(graph, p, mergedIdentities);

          const std::array<std::size_t, 2> classes{0, static_cast<std::size_t>(memberClass) + 1};
          const double regionEta = caloEntryEta[p] != truth::kNoCaloEntry ? caloEntryEta[p] : x[kEta];
          for (const std::size_t cl : classes) {
            for (const std::size_t r : slotsOf(std::abs(regionEta))) {
              auto const& f = histograms.truth[truthIndex(k, w, l, cl, r)];
              auto fill = [&](TruthCount n) {
                for (int v = 0; v < kNVariables; ++v)
                  if ((failed & ~cutBits_[v]) == 0u)
                    f.counts[n][v]->Fill(x[v]);
              };
              fill(kSimul);
              if (outcome.reconstructed)
                fill(kAssoc);
              if (outcome.trackCorrect)
                fill(kAssocTrack);
              if (outcome.pidCorrect)
                fill(kAssocPid);
              if (outcome.energyCorrect)
                fill(kAssocEnergy);
              if (outcome.multiple)
                fill(kMultiple);
              if (merged)
                fill(kMerged);
              if (failed != 0u)
                continue;
              if (f.nCandidates != nullptr)
                f.nCandidates->Fill(std::min<std::size_t>(onMember.size(), kNCandidateBins - 1));
              if (!outcome.reconstructed || p4.energy() <= 0. || !hasDirection)
                continue;
              auto const& rep = candidates[belonging[t][outcome.representative]];
              const double e = p4.energy();
              f.residuals[kEresVsEnergy]->Fill(e, (rep.energy() - e) / e);
              f.residuals[kEresVsEta]->Fill(p4.eta(), (rep.energy() - e) / e);
              f.residuals[kRawEresVsEnergy]->Fill(e, (rep.rawEnergy() - e) / e);
              f.residuals[kEsumresVsEnergy]->Fill(e, (outcome.energySum - e) / e);
              f.residuals[kPtresVsPt]->Fill(p4.pt(), (rep.pt() - p4.pt()) / p4.pt());
              f.residuals[kEtaresVsEta]->Fill(p4.eta(), rep.eta() - p4.eta());
              f.residuals[kPhiresVsEta]->Fill(p4.eta(), reco::deltaPhi(rep.phi(), p4.phi()));
              if (f.confusion != nullptr)
                f.confusion->Fill(static_cast<double>(memberClass) + 0.5,
                                  static_cast<double>(info[belonging[t][outcome.representative]].cls) + 0.5);
            }
          }
        }
      }
    }
  }
}

void TruthBranchCandidateValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<std::string>("dirName", "TruthInfo/Offline/Candidates/");
  desc.add<std::vector<edm::InputTag>>("recoCollections", {edm::InputTag("ticlCandidate")});
  desc.add<std::string>("associator", "truthBranchCandidateAssociators");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"))
      ->setComment("The collection the candidate tracks point to, as the track associator keys it");
  desc.add<std::string>("trackAssociator", "allTrackToTruthBranchAssociators");
  desc.add<std::string>("targetsProducer", "truthBranchTargets")
      ->setComment("Module holding the level members, their eligibility and the assignable roots");
  desc.add<std::vector<std::string>>("workingPoints", {"AdaptiveTight", "AdaptiveNominal"})
      ->setComment("The working points whose calorimeter match is the identity of a candidate, one folder each");
  desc.add<std::string>("referenceWorkingPoint", "Fixed")
      ->setComment(
          "The working point whose map keeps every candidate branch: dominance is read from it, and the particle of "
          "a track is the first entry of its track map row when that is an assignable root");
  desc.add<std::vector<std::string>>("truthLevels", {"reconstructableFromSignal", "caloBoundary"})
      ->setComment("The levels whose members are the efficiency denominators, one folder each");
  desc.add<bool>("requireHits", false)
      ->setComment(
          "Keep in the truth-driven denominators only the members whose subgraph has a calorimeter hit in "
          "denominatorDetectors");
  desc.add<std::vector<std::string>>("denominatorDetectors", {})
      ->setComment(
          "DetId::Detector names of the requireHits cut: the denominatorDetectors of the associator. Empty means the "
          "whole calorimeter channel");
  desc.add<std::string>("dominanceLevel", "caloBoundary")
      ->setComment("The level the fake criterion and the reco-driven class check project on; one of truthLevels");
  desc.add<double>("minLeadingTruthShare", 0.5)
      ->setComment("A candidate is not a fake when one dominance-level member owns at least this share of it");
  desc.add<double>("minTrackPurity", 0.75)
      ->setComment(
          "A track has a particle when at least this fraction of its hits belong to it, the Cut_RecoToSim of "
          "quickTrackAssociatorByHits");
  desc.add<double>("energyResponseMin", 0.7)->setComment("Lower edge of the energy step of the ladder, E/E_true");
  desc.add<double>("energyResponseMax", 1.3)->setComment("Upper edge of the energy step of the ladder, E/E_true");

  edm::ParameterSetDescription block;
  const std::vector<std::tuple<std::string, int, double, double, double>> axes{{"pt", 50, 0., 1000., 0.1},
                                                                               {"eta", 50, -4.5, 4.5, 0.},
                                                                               {"phi", 36, -3.2, 3.2, 0.},
                                                                               {"energy", 50, 0., 2000., 1.}};
  for (auto const& [name, nbins, lo, hi, linthresh] : axes) {
    block.add<int>("nint_" + name, nbins);
    block.add<double>("min_" + name, lo);
    block.add<double>("max_" + name, hi);
    block.add<double>("linthresh_" + name, linthresh)->setComment("0 for a uniform axis, else symlog from here");
  }
  const std::vector<std::tuple<std::string, int, double, double>> residualAxes{
      {"res", 100, -1., 1.}, {"res_energy", 25, 0., 500.}, {"res_eta", 20, -4., 4.}, {"res_pt", 15, 0., 150.}};
  for (auto const& [name, nbins, lo, hi] : residualAxes) {
    block.add<int>("nint_" + name, nbins);
    block.add<double>("min_" + name, lo);
    block.add<double>("max_" + name, hi);
  }
  block.add<std::vector<std::string>>("etaRegions", {"eta15to30"})
      ->setComment("Absolute pseudorapidity bands booked as sub-folders next to the inclusive folder");
  desc.add<edm::ParameterSetDescription>("histoBlock", block);
  descriptions.add("truthBranchCandidateValidator", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthBranchCandidateValidator);
