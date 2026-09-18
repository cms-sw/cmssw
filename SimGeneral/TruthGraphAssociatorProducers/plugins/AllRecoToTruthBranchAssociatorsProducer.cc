// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

// One producer, every configured reco collection of one type, every branch-association
// working point. Follows the All* pattern of
// SimCalorimetry/HGCalAssociatorProducers: the module takes a VInputTag of reco
// collections and emits one pair of association maps per (collection, working point),
// with instance labels derived from the input tags.
//
// The reco type only has to be adaptable to (DetId, fraction) hits. Which adapter
// applies is decided by a concept rather than by a per-type producer, so a new domain
// is a truth::recoHits overload plus a label in truthGraphAssociationLabels_cff, not a
// new plugin.
//
// Working points differ only in the arguments passed to bestAdaptiveBranch, not in the
// associator itself, so the inverted DetId index is built ONCE per event and reused
// across every working point.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <concepts>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <string_view>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/AssignableTarget.h"
#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/Interactions.h"
#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"
#include "PhysicsTools/TruthInfo/interface/TrackerCells.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

// The producer template below is a member of this namespace rather than of the anonymous
// one, and it holds a VertexResolution. A type with internal linkage cannot be the type of
// a member of a class with external linkage, so this enum needs a named home.
namespace truthassociation {
  //
  //   Immediate    the production vertex of the matched particle itself. Right for a
  //                secondary vertex, which IS a decay or interaction vertex: the tracks
  //                that belong to it were produced there.
  //   Interaction  the one vertex representing the interaction the particle belongs to,
  //                so a track from a decay downstream of the vertex is counted at the
  //                vertex the chain started from. Right for a primary vertex, where the
  //                question is which interaction a track came from, not which decay.
  enum class VertexResolution { Immediate, Interaction };
}  // namespace truthassociation

namespace {
  using truth::byAscendingScore;
  using truthassociation::VertexResolution;

  // DetId::Detector by name, so the shared-energy denominator is a readable configuration
  // list instead of an integer mask. An unknown name is a configuration error.
  DetId::Detector detectorFromName(std::string const& name) {
    static constexpr std::pair<std::string_view, DetId::Detector> kDetectors[] = {
        {"Tracker", DetId::Tracker},
        {"Muon", DetId::Muon},
        {"Ecal", DetId::Ecal},
        {"Hcal", DetId::Hcal},
        {"Calo", DetId::Calo},
        {"Forward", DetId::Forward},
        {"VeryForward", DetId::VeryForward},
        {"HGCalEE", DetId::HGCalEE},
        {"HGCalHSi", DetId::HGCalHSi},
        {"HGCalHSc", DetId::HGCalHSc},
        {"HGCalTrigger", DetId::HGCalTrigger}};
    for (auto const& [known, detector] : kDetectors) {
      if (known == name) {
        return detector;
      }
    }
    throw cms::Exception("Configuration") << "denominatorDetectors: unknown DetId::Detector name '" << name << "'";
  }

  // A reco type that yields its own hits needs nothing but itself.
  template <typename RECO>
  concept SelfContainedRecoHits = requires(RECO const& r) {
    { truth::recoHits(r) } -> std::same_as<std::vector<truth::RecoHit>>;
  };

  // A reco type built out of layer clusters needs the layer-cluster collection too.
  template <typename RECO>
  concept LayerClusterBackedRecoHits = requires(RECO const& r, std::vector<reco::CaloCluster> const& lcs) {
    { truth::recoHits(r, lcs) } -> std::same_as<std::vector<truth::RecoHit>>;
  };

  template <typename RECO>
  concept AdaptableToTruthHits = SelfContainedRecoHits<RECO> || LayerClusterBackedRecoHits<RECO>;

  // How a domain reaches the truth is a property of its reco type, not of runtime
  // configuration. Two strategies cover everything:
  //
  //   HitBased         the object owns detector hits, so it is matched directly
  //                    (tracks by shared hits, tracksters by shared energy).
  //   ConstituentBased the object is BUILT from objects that are already associated,
  //                    so its truth is aggregated from theirs rather than recomputed
  //                    from hits. A vertex shares tracks, a jet shares constituents,
  //                    a candidate shares a track and clusters. This is the layering
  //                    CMSSW already uses: VertexAssociatorByPositionAndTracks
  //                    consumes the track maps, it does not revisit hits.
  //
  // Binding payload and strategy to the type means the declared product type and the
  // produced one cannot drift apart.
  enum class AssociationStrategy { HitBased, ConstituentBased };

  template <typename RECO>
  struct TruthAssociationTraits;

  template <>
  struct TruthAssociationTraits<reco::Track> {
    static constexpr auto strategy = AssociationStrategy::HitBased;
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr truth::HitChannel channel = truth::HitChannel::Tracker;
    static constexpr auto metric = truth::BranchHitAssociator::Metric::SharedHits;
    static constexpr const char* cfiName = "allTrackToTruthBranchAssociators";
  };

  // A vertex carries no hits of its own: its truth is whatever its tracks point to.
  // The payload is therefore a FRACTION of the vertex's tracks, weighted the way
  // calculateVertexSharedTracks weights them, not an energy.
  template <>
  struct TruthAssociationTraits<reco::Vertex> {
    static constexpr auto strategy = AssociationStrategy::ConstituentBased;
    using ConstituentType = reco::Track;
    using MapType = ticl::TICLAssociationMap<ticl::mapWithFractionAndScore>;
    static constexpr const char* cfiName = "allVertexToTruthBranchAssociators";

    // Visit (constituent index into its own collection, weight). The index is the Ref
    // key, which is exactly the row the constituent's association map is indexed by.
    //
    // The weight is pt SQUARED, which is what CMSSW's own vertex association uses:
    // calculateVertexSharedTracks returns sharedPt2Fraction as
    // sum(pt^2 of shared tracks) / sum(pt^2 of ALL the vertex's tracks)
    // (SimTracker/VertexAssociation/src/calculateVertexSharedTracks.cc). The vertex FIT
    // weight answers a different question: it says how strongly a track constrained the
    // fit, not how much of the vertex's momentum it carries, and it gives a soft pileup
    // track the same standing as a hard signal one.
    template <typename F>
    static void forEachConstituent(reco::Vertex const& vertex, F&& visit) {
      for (auto it = vertex.tracks_begin(); it != vertex.tracks_end(); ++it) {
        const float pt = (*it)->pt();
        visit(static_cast<unsigned int>(it->key()), pt * pt);
      }
    }

    static float totalWeight(reco::Vertex const& vertex) {
      float total = 0.f;
      forEachConstituent(vertex, [&total](unsigned int, float w) { total += w; });
      return total;
    }
  };

  // A particle-flow cluster owns its calorimeter cells directly: reco::PFCluster derives
  // from reco::CaloCluster and addRecHitFraction fills the base hitsAndFractions, so the
  // reco::CaloCluster adapter in RecoHitAdapters.h applies unchanged and the cluster is
  // matched on shared energy like a trackster. One PFCluster type serves every block
  // element flavour (ECAL, HCAL, HO, HF, PS, HGCAL): the importer picks the enum from
  // the cluster's layer, the object is the same. Which detector the shared-energy
  // denominator covers is configuration (denominatorDetectors), one module per
  // subdetector, because the efficiency gate is the branch energy fraction there and a
  // hadron branch scored against a joint ECAL+HCAL denominator can never reach the
  // individual threshold with an ECAL cluster alone.
  template <>
  struct TruthAssociationTraits<reco::PFCluster> {
    static constexpr auto strategy = AssociationStrategy::HitBased;
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr truth::HitChannel channel = truth::HitChannel::Calo;
    static constexpr auto metric = truth::BranchHitAssociator::Metric::SharedEnergy;
    static constexpr const char* cfiName = "truthBranchPFClusterAssociators";
  };

  // A trackster owns calorimeter energy through its layer clusters, so it is matched
  // directly like a track, but on SHARED ENERGY in the calorimeter channel rather than
  // on a hit count in the tracker. This is the same metric the TICL trackster
  // validation scores against, so the two are comparable.
  template <>
  struct TruthAssociationTraits<ticl::Trackster> {
    static constexpr auto strategy = AssociationStrategy::HitBased;
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr truth::HitChannel channel = truth::HitChannel::Calo;
    static constexpr auto metric = truth::BranchHitAssociator::Metric::SharedEnergy;
    static constexpr const char* cfiName = "truthBranchTracksterAssociators";
  };

  [[nodiscard]] inline std::optional<uint32_t> countingVertex(
      truth::Graph const& graph,
      uint32_t particleId,
      VertexResolution resolution,
      std::unordered_map<uint64_t, uint32_t> const& interactionVertex) {
    if (resolution == VertexResolution::Interaction) {
      const auto it = interactionVertex.find(graph.particles()[particleId].eventId);
      if (it == interactionVertex.end()) {
        return std::nullopt;
      }
      return it->second;
    }
    const auto production = truth::Particle(&graph, particleId).productionVertices();
    if (production.empty()) {
      return std::nullopt;
    }
    return production.front().id();
  }

  template <typename RECO>
  concept HitBasedDomain = TruthAssociationTraits<RECO>::strategy == AssociationStrategy::HitBased;

  template <typename RECO>
  concept ConstituentBasedDomain = TruthAssociationTraits<RECO>::strategy == AssociationStrategy::ConstituentBased;
}  // namespace

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
class AllRecoToTruthBranchAssociatorsProducer : public edm::global::EDProducer<> {
public:
  explicit AllRecoToTruthBranchAssociatorsProducer(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  struct WorkingPoint {
    std::string name;
    float reverseWeight;
    float maxReverseScore;
    bool adaptive;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  // Calorimetric domains only: the rechit collections whose energies weight every
  // cell of the shared-energy metric, as the TICL associators weight them.
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;
  // One warning per job when no rechit collection is present, because the metric
  // then falls back to sim-energy weights and its scores are not the TICL ones.
  mutable std::once_flag recHitsWarned_;
  mutable std::once_flag moduleKeyedWarned_;
  mutable std::once_flag rowOutOfRangeWarned_;
  mutable std::once_flag placeholderVertexWarned_;

  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<RECO>>>> recoTokens_;
  // One warning per collection per job when its input is absent: a silently empty map
  // is indistinguishable from a perfectly working associator on a bad label, and the
  // downstream symptom is a fully booked, zero-entry DQM folder.
  mutable std::vector<std::once_flag> missingWarned_;
  // One warning per collection per job when none of its hits is in the configured
  // denominator scope, because every shared-energy fraction is then zero, which reads
  // like a reconstruction that matches nothing. A barrel trackster collection under an
  // endcap-only scope is that case.
  mutable std::vector<std::once_flag> outOfScopeWarned_;
  std::vector<WorkingPoint> workingPoints_;
  // The detectors the sim-normalised shared-energy fraction is normalised to. It is
  // configuration, so one truth branch is scored against the same denominator in every
  // event and in an event where the collection is empty.
  uint32_t denominatorDetectors_ = truth::BranchHitAssociator::kAllDetectors;
  const bool truthToRecoSignalOnly_;
  const bool heavyFlavorOnly_;
  // Composite domains only: the worst score a constituent's best match may have and
  // still place the constituent at a truth vertex.
  float maxConstituentScore_ = 1.f;
  // The selector-passing candidate roots, computed once per event by the shared
  // TruthBranchTargetsProducer together with the level denominators and signal seeds.
  edm::EDGetTokenT<std::vector<unsigned int>> targetsToken_;
  // The subset of those roots a reco object may be assigned to, from the same producer.
  edm::EDGetTokenT<std::vector<unsigned int>> assignableTargetsToken_;

  using Traits = TruthAssociationTraits<RECO>;
  using MapType = typename Traits::MapType;

  // Composite domains read their constituents' association maps instead of hits. The
  // upstream module is named by a cms.string and the instance labels are rebuilt here,
  // the same way the HGCal All* producers reach allHitToTracksterAssociations.
  // Constituents are tracks for every composite domain shipped here; a future
  // non-track constituent belongs in the domain's own trait, not in a conditional.
  using ConstituentMapType = TruthAssociationTraits<reco::Track>::MapType;
  std::vector<std::vector<edm::EDGetTokenT<ConstituentMapType>>> constituentMapTokens_;
  VertexResolution vertexResolution_ = VertexResolution::Immediate;
};

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
AllRecoToTruthBranchAssociatorsProducer<RECO>::AllRecoToTruthBranchAssociatorsProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      truthToRecoSignalOnly_(cfg.getParameter<bool>("truthToRecoSignalOnly")),
      heavyFlavorOnly_(cfg.getParameter<bool>("heavyFlavorOnly")) {
  if constexpr (LayerClusterBackedRecoHits<RECO>) {
    layerClustersToken_ = consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"));
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits"))
      hgcalRecHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("pfRecHits"))
      pfRecHitTokens_.push_back(consumes<reco::PFRecHitCollection>(tag));
  }

  targetsToken_ = consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("targetsSrc"));
  assignableTargetsToken_ =
      consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("assignableTargetsSrc"));

  if constexpr (ConstituentBasedDomain<RECO>) {
    maxConstituentScore_ = cfg.getParameter<double>("maxConstituentScore");
  }

  if (auto const detectors = cfg.getParameter<std::vector<std::string>>("denominatorDetectors"); !detectors.empty()) {
    denominatorDetectors_ = 0u;
    for (auto const& name : detectors) {
      denominatorDetectors_ |= 1u << static_cast<uint32_t>(detectorFromName(name));
    }
  }

  const auto names = cfg.getParameter<std::vector<std::string>>("workingPointNames");
  const auto weights = cfg.getParameter<std::vector<float>>("adaptiveReverseWeight");
  const auto ceilings = cfg.getParameter<std::vector<float>>("adaptiveMaxReverseScore");
  if (names.size() != weights.size() || names.size() != ceilings.size()) {
    throw cms::Exception("Configuration")
        << "workingPointNames, adaptiveReverseWeight and adaptiveMaxReverseScore must have the same length";
  }
  if (names.empty()) {
    throw cms::Exception("Configuration")
        << "workingPointNames is empty: the truth-driven maps are filled inside the working-point loop, so an empty "
           "list would silently produce empty TruthToReco products";
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    // "Fixed" means the plain per-root match; every other point drives the climb.
    workingPoints_.push_back({names[i], weights[i], ceilings[i], names[i] != "Fixed"});
  }

  if constexpr (ConstituentBasedDomain<RECO>) {
    // A composite object's truth target is a vertex, not a branch at some level, so
    // there is a single denominator.
    produces<std::vector<unsigned int>>("truthToRecoTargets");
    // A composite object is associated to a truth VERTEX, so its efficiency denominator
    // is a set of vertices, not of branch roots.
    produces<std::vector<unsigned int>>("selectedTruthVertices");
    const auto resolution = cfg.getParameter<std::string>("vertexResolution");
    if (resolution == "interaction") {
      vertexResolution_ = VertexResolution::Interaction;
    } else if (resolution == "immediate") {
      vertexResolution_ = VertexResolution::Immediate;
    } else {
      throw cms::Exception("Configuration")
          << "vertexResolution must be 'immediate' or 'interaction', got '" << resolution << "'";
    }
  }

  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recoCollections")) {
    // Key rule for this package: label and instance joined by an underscore, the same
    // string that names the DQM folder. HGCal concatenates for products and
    // underscores for folders; keeping one rule avoids that asymmetry.
    std::string key = tag.label();
    if (!tag.instance().empty()) {
      key += "_" + tag.instance();
    }
    recoTokens_.emplace_back(key, consumes<std::vector<RECO>>(tag));

    if constexpr (ConstituentBasedDomain<RECO>) {
      // One constituent map per working point, in the same order as workingPoints_.
      const auto upstream = cfg.getParameter<std::string>("constituentAssociator");
      const auto constituentKey = cfg.getParameter<std::string>("constituentCollection");
      std::vector<edm::EDGetTokenT<ConstituentMapType>> perWp;
      perWp.reserve(workingPoints_.size());
      for (auto const& wp : workingPoints_) {
        perWp.push_back(
            consumes<ConstituentMapType>(edm::InputTag(upstream, constituentKey + "RecoToTruth" + wp.name)));
      }
      constituentMapTokens_.push_back(std::move(perWp));
    }

    // The two directions are NOT transposes of each other and are deliberately not
    // named as if they were.
    //
    // RecoToTruth is reco-driven: given a reco object, the adaptive search picks the
    // graph level that best matches it, so there is one product per working point. Its
    // score is 1 - RECO purity, the reco object being the denominator.
    //
    // TruthToReco is truth-driven: the truth target is fixed A PRIORI by the domain's
    // resolution, so there is ONE product. The reco side of each pair still comes from
    // a matching pass, which runs at the FIRST listed working point; the shipped config
    // lists "Fixed" first. Its score is 1 - TRUTH purity, the truth object being the
    // denominator.
    for (auto const& wp : workingPoints_) {
      produces<MapType>(key + "RecoToTruth" + wp.name);
    }
    produces<MapType>(key + "TruthToReco");
  }
  missingWarned_ = std::vector<std::once_flag>(recoTokens_.size());
  outOfScopeWarned_ = std::vector<std::once_flag>(recoTokens_.size());
}

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
void AllRecoToTruthBranchAssociatorsProducer<RECO>::produce(edm::StreamID,
                                                            edm::Event& event,
                                                            edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  if constexpr (!ConstituentBasedDomain<RECO>) {
    if (Traits::channel == truth::HitChannel::Tracker && truth::isModuleKeyedTracker(hitIndex)) {
      std::call_once(moduleKeyedWarned_, [] {
        edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
            << "the tracker truth of the input carries no cells, so no track matches it. The input was "
               "digitised before the tracker truth was keyed by cell; reprocess it from the DIGI step.";
      });
    }
  }

  std::vector<reco::CaloCluster> const* layerClusters = nullptr;
  // The rechit energy of every cell, the weight of the shared-energy metric. Absent
  // collections are skipped; an empty table makes the associator weight cells by their
  // sim energy instead, so the job still produces maps. The barrel PFRecHits (DetId
  // detectors 3 and 4) go in before the HGCAL rechits (8 to 10), and each collection is
  // sorted by DetId, so the table is built in order and finalize() does not sort.
  truth::CellEnergyTable recHitEnergies;
  truth::CellEnergyTable const* recHitEnergiesPtr = nullptr;
  if constexpr (LayerClusterBackedRecoHits<RECO>) {
    for (auto const& token : pfRecHitTokens_) {
      const edm::Handle<reco::PFRecHitCollection> handle = event.getHandle(token);
      if (!handle.isValid())
        continue;
      recHitEnergies.reserve(recHitEnergies.size() + handle->size());
      for (auto const& hit : *handle)
        recHitEnergies.add(hit.detId(), hit.energy());
    }
    for (auto const& token : hgcalRecHitTokens_) {
      const edm::Handle<HGCRecHitCollection> handle = event.getHandle(token);
      if (!handle.isValid())
        continue;
      recHitEnergies.reserve(recHitEnergies.size() + handle->size());
      for (auto const& hit : *handle)
        recHitEnergies.add(hit.id().rawId(), hit.energy());
    }
    recHitEnergies.finalize();
    if (recHitEnergies.empty()) {
      std::call_once(recHitsWarned_, [] {
        edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
            << "no rechit collection present; the shared-energy metric weights cells by sim energy";
      });
    } else {
      recHitEnergiesPtr = &recHitEnergies;
    }
    // Tolerant like the reco collections below: the HLT twin runs in jobs whose input
    // may carry no HLT reconstruction at all, and then it must produce empty maps
    // rather than throw. Trackster collections cannot be adapted without the clusters,
    // so they are skipped when the clusters are absent.
    const edm::Handle<std::vector<reco::CaloCluster>> handle = event.getHandle(layerClustersToken_);
    if (handle.isValid()) {
      layerClusters = &(*handle);
    } else {
      edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
          << "layer clusters absent; trackster collections will produce empty maps this event";
    }
  }

  const unsigned int nBranches = graph.nParticles();

  // The selector-passing candidate roots, computed once per event by the shared
  // TruthBranchTargetsProducer alongside the level denominators and signal seeds.
  auto const& selectedRoots = event.get(targetsToken_);
  // Membership test for the adaptive answer below, one lookup per candidate. The product
  // is a sorted id list like every other target product; the mask is the per-event
  // expansion of it. A composite domain answers with a vertex, so it needs no mask.
  const bool anyAdaptiveWorkingPoint =
      std::any_of(workingPoints_.begin(), workingPoints_.end(), [](WorkingPoint const& wp) { return wp.adaptive; });
  std::vector<uint8_t> isAssignable;
  if constexpr (!ConstituentBasedDomain<RECO>) {
    if (anyAdaptiveWorkingPoint) {
      isAssignable.assign(nBranches, 0);
      for (const unsigned int id : event.get(assignableTargetsToken_)) {
        if (id < isAssignable.size()) {
          isAssignable[id] = 1;
        }
      }
    }
  }

  // The denominator of the truth-side fraction: what each truth vertex itself produced,
  // in the same pt^2 weighting the numerator uses.
  std::unordered_map<unsigned int, float> truthWeightPerVertex;

  unsigned int placeholderCount = 0;
  std::unordered_map<uint64_t, uint32_t> interactionVertex;
  if (ConstituentBasedDomain<RECO> && vertexResolution_ == VertexResolution::Interaction) {
    for (auto const& interaction : truth::interactions(graph)) {
      interactionVertex.emplace(interaction.eventId(), interaction.vertexId());
      placeholderCount += interaction.isPlaceholder() ? 1 : 0;
    }
  }
  if (placeholderCount > 0) {
    std::call_once(placeholderVertexWarned_, [placeholderCount] {
      edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
          << placeholderCount
          << " interactions resolve only to a logical vertex that did not merge with a SimVertex and whose "
             "position is indistinguishable from a default-constructed one. Their constituents are counted "
             "there, so a vertex efficiency or purity for those interactions carries no position. This is what "
             "a pileup sub-event looks like when all of its GenToSim links were dropped. Reported once per job.";
    });
  }

  if constexpr (ConstituentBasedDomain<RECO>) {
    // The vertices a composite object could have been reconstructed at: those where at
    // least two findable tracks were produced. One track cannot make a vertex, so a
    // one-particle vertex in the denominator is a guaranteed miss that scales every
    // efficiency down without measuring anything, the same reason the branch selector
    // guards the particle denominator. The count, the signal count and the pt^2 weight
    // all run over ONE population: in-time, charged, one entry per physical particle.
    // Counting the gate over a different population than the weight readmits the
    // one-track vertex through a neutral member or a root's own selected ancestor.
    // Charged, because the constituents are tracks and a neutrino carries pt^2 no
    // vertex finder can recover. One entry per physical particle: with Interaction
    // resolution a whole decay chain resolves to one vertex, so a tau and its three
    // prongs would all enter unless the candidates are reduced to their deepest
    // antichain first. Immediate resolution needs no reduction, its members being one
    // vertex's outgoing particles.
    std::unordered_map<unsigned int, unsigned int> rootsPerVertex;
    std::unordered_map<unsigned int, unsigned int> signalRootsPerVertex;
    {
      std::vector<uint32_t> counted = selectedRoots;
      if (vertexResolution_ == VertexResolution::Interaction) {
        truth::dropCoveredMembers(graph, counted, /*keepDeepest=*/true);
      }
      for (uint32_t root : counted) {
        // In-time only, as the reference vertex validation counts only bunch-crossing-0
        // simulated vertices in its denominator
        // (Validation/RecoVertex/src/PrimaryVertexAnalyzer4PUSlimmed.cc:877-883).
        if (!truth::Branch(&graph, root).isInTime()) {
          continue;
        }
        if (graph.particle(root).charge() == 0) {
          continue;
        }
        // Same resolution the numerator uses. A denominator counted at a different set
        // of vertices than the numerator measures nothing.
        if (const auto vertexId = countingVertex(graph, root, vertexResolution_, interactionVertex)) {
          ++rootsPerVertex[*vertexId];
          if (graph.particles()[root].isSignal()) {
            ++signalRootsPerVertex[*vertexId];
          }
          const float rootPt = static_cast<float>(graph.particles()[root].momentum.pt());
          truthWeightPerVertex[*vertexId] += rootPt * rootPt;
        }
      }
    }

    // The denominator is where a heavy-flavour hadron decayed, which is what a secondary
    // vertex is: inclusiveSecondaryVertices reconstructs displaced heavy-flavour
    // vertices, not every nuclear interaction, conversion and decay in flight. The levels
    // are antichains, so a B* radiating down to a B contributes one vertex rather than one
    // per generator copy. Beauty and charm are asked separately because a B decays to a D
    // and a combined level would drop every charm vertex.
    const std::unordered_set<unsigned int> heavyFlavorDecayVertices = [&graph, heavyFlavorOnly = heavyFlavorOnly_] {
      std::unordered_set<unsigned int> vertices;
      // Only the secondary-vertex flavour of this producer reads the set.
      if (!heavyFlavorOnly)
        return vertices;
      for (const truth::Level level : {truth::Level::BHadrons, truth::Level::CHadrons}) {
        for (const uint32_t id : truth::levelAntichain(graph, level)) {
          for (const uint32_t vertexId : graph.decayVertices(id)) {
            vertices.insert(vertexId);
          }
        }
      }
      return vertices;
    }();

    auto selectedVertices = std::make_unique<std::vector<unsigned int>>();
    auto targets = std::make_unique<std::vector<unsigned int>>();
    for (auto const& [vertexId, count] : rootsPerVertex) {
      if (count < 2u) {
        continue;
      }
      // Junk-vertex guard of the reference vertex validation: a simulated vertex
      // beyond |z| of 1000 cm is not counted
      // (Validation/RecoVertex/src/PrimaryVertexAnalyzer4PUSlimmed.cc:885-886).
      if (std::abs(graph.vertices()[vertexId].position.z()) > 1000.) {
        continue;
      }
      if (heavyFlavorOnly_ && heavyFlavorDecayVertices.count(vertexId) == 0u) {
        continue;
      }
      selectedVertices->push_back(vertexId);
      // Signal is decided from the PARTICLES produced there, not from the vertex's own
      // eventId: a collapsed GEN vertex carries 0 even when everything it produced
      // belongs to a pileup interaction.
      if (!truthToRecoSignalOnly_ || signalRootsPerVertex[vertexId] > 0u) {
        targets->push_back(vertexId);
      }
    }
    std::sort(selectedVertices->begin(), selectedVertices->end());
    std::sort(targets->begin(), targets->end());
    event.put(std::move(selectedVertices), "selectedTruthVertices");
    event.put(std::move(targets), "truthToRecoTargets");
  }

  // Associator cache shared by all collections of this domain, keyed by mask.
  std::vector<std::pair<uint32_t, std::unique_ptr<truth::BranchHitAssociator>>> associatorPerMask;

  for (std::size_t collectionIndex = 0; collectionIndex < recoTokens_.size(); ++collectionIndex) {
    auto const& [key, token] = recoTokens_[collectionIndex];
    edm::Handle<std::vector<RECO>> handle;
    event.getByToken(token, handle);
    // A trackster collection without its layer clusters cannot be adapted to hits, so
    // it is treated exactly like an absent collection: valid empty maps.
    bool valid = handle.isValid();
    if constexpr (LayerClusterBackedRecoHits<RECO>) {
      valid = valid && layerClusters != nullptr;
    }
    const unsigned int nReco = valid ? handle->size() : 0u;
    if (!valid) {
      std::call_once(missingWarned_[collectionIndex], [&key] {
        edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
            << "input collection '" << key << "' absent; its association maps will be empty for this job";
      });
    }

    // Truth-driven direction, built ONCE: the truth target is fixed a priori, so the
    // reco-driven working point plays no part in it. Its score is 1 - truth purity.
    const unsigned int nTruthRows = ConstituentBasedDomain<RECO> ? graph.nVertices() : nBranches;
    auto truthToReco = std::make_unique<MapType>(nTruthRows);

    // Hit-based domains: each object's hit adaptation is independent of the working
    // point, so it is built ONCE per collection and shared by every working point below.
    std::vector<std::vector<truth::RecoHit>> recoHitsPerObject;
    if constexpr (!ConstituentBasedDomain<RECO>) {
      recoHitsPerObject.resize(nReco);
      for (unsigned int i = 0; i < nReco; ++i) {
        if constexpr (LayerClusterBackedRecoHits<RECO>) {
          recoHitsPerObject[i] = truth::recoHits((*handle)[i], *layerClusters);
        } else {
          recoHitsPerObject[i] = truth::recoHits((*handle)[i]);
        }
      }
      if constexpr (Traits::metric == truth::BranchHitAssociator::Metric::SharedEnergy) {
        uint32_t seen = 0;
        for (auto const& hits : recoHitsPerObject) {
          for (auto const& hit : hits) {
            seen |= truth::BranchHitAssociator::detectorBit(hit.detId);
          }
        }
        if (seen != 0u && (seen & denominatorDetectors_) == 0u) {
          std::call_once(outOfScopeWarned_[collectionIndex], [&key] {
            edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
                << "collection '" << key
                << "' has no hit in the configured denominatorDetectors; every shared-energy fraction will be zero";
          });
        }
      }
    }

    // Composite domains only: (reco index, shared weight) per truth vertex and the
    // per-truth-vertex total, so the truth-normalised fraction can be formed once every
    // reco object of the collection has contributed.
    std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, float>>> sharedWeightPerTruthVertex;

    // Hit-based domains: the associator depends on the graph, the selected roots and
    // the detector mask, never on the reco collection itself, so it is cached per
    // mask; collections of one domain produce the same mask, giving one build.
    truth::BranchHitAssociator const* hitAssociator = nullptr;
    if constexpr (!ConstituentBasedDomain<RECO>) {
      for (auto const& [mask, cached] : associatorPerMask) {
        if (mask == denominatorDetectors_) {
          hitAssociator = cached.get();
          break;
        }
      }
      if (hitAssociator == nullptr) {
        associatorPerMask.emplace_back(denominatorDetectors_,
                                       std::make_unique<truth::BranchHitAssociator>(hitIndex,
                                                                                    selectedRoots,
                                                                                    Traits::metric,
                                                                                    Traits::channel,
                                                                                    /*emptyRootsMeansAll=*/false,
                                                                                    denominatorDetectors_,
                                                                                    recHitEnergiesPtr));
        hitAssociator = associatorPerMask.back().second.get();
      }
    }

    if constexpr (ConstituentBasedDomain<RECO>) {
      for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
        auto const& wp = workingPoints_[wpIndex];
        auto recoToTruth = std::make_unique<MapType>(nReco);

        // A composite object is associated to a truth VERTEX, not to a particle branch.
        // Keying the aggregation by the branch a constituent points at cannot disagree
        // with itself, so every object matched something and the purity was 1 by
        // construction. Keying it by the PRODUCTION VERTEX of that branch is what makes
        // the number mean anything: constituents whose particles were produced at an
        // unrelated vertex are contamination, and the leading vertex's share is the
        // purity.
        auto const& constituentMap = event.get(constituentMapTokens_[collectionIndex][wpIndex]);
        for (unsigned int i = 0; i < nReco; ++i) {
          auto const& object = (*handle)[i];
          // Summed in its own pass, not fused into the scan below: fusing changes the
          // inlining context of the float accumulation and with it the rounding of the
          // pt^2 sums, which moves association scores in the last ulp.
          const float total = Traits::totalWeight(object);
          if (total <= 0.f) {
            continue;
          }
          std::unordered_map<unsigned int, float> weightPerVertex;
          Traits::forEachConstituent(object, [&](unsigned int constituentIndex, float weight) {
            if (constituentIndex >= constituentMap.size()) {
              return;
            }
            // maps are score-sorted, so [0] is the constituent's best match
            for (auto const& match : constituentMap[constituentIndex]) {
              // A constituent donates its whole weight to the vertex it points at, so a
              // weak match must not point anywhere. 1 - score is the constituent's reco
              // purity, the same quantity the tracker association thresholds.
              if (match.score() > maxConstituentScore_) {
                break;
              }
              const unsigned int particle = match.index();
              if (particle < nBranches) {
                if (const auto vertexId = countingVertex(graph, particle, vertexResolution_, interactionVertex)) {
                  weightPerVertex[*vertexId] += weight;
                }
              }
              break;
            }
          });
          // Denominator over ALL constituents, the CMSSW convention: a track with no
          // truth match lowers the shared fraction. The pt^2 weighting keeps that
          // dilution small, because the unmatched tracks are the soft ones.
          for (auto const& [vertexId, weight] : weightPerVertex) {
            // RECO purity: the leading truth vertex's share of THIS reco object's pt^2.
            const float recoPurity = weight / total;
            recoToTruth->insert(i, vertexId, recoPurity, 1.f - recoPurity);
            // TRUTH purity: the shared weight over what the truth vertex produced,
            // formed below once the whole collection has been seen.
            if (wpIndex == 0) {
              sharedWeightPerTruthVertex[vertexId].emplace_back(i, weight);
            }
          }
        }

        if (wpIndex == 0) {
          for (auto const& [vertexId, entries] : sharedWeightPerTruthVertex) {
            auto const denominatorIt = truthWeightPerVertex.find(vertexId);
            if (denominatorIt == truthWeightPerVertex.end() || denominatorIt->second <= 0.f) {
              continue;
            }
            const float denominator = denominatorIt->second;
            for (auto const& [recoIndex, weight] : entries) {
              // A reco vertex can hold a track the truth vertex did not produce, so the
              // ratio is clamped.
              const float truthPurity = std::min(1.f, weight / denominator);
              truthToReco->insert(vertexId, recoIndex, truthPurity, 1.f - truthPurity);
            }
          }
        }

        // Ascending score, so [0] is the best match; consumers rely on this. An explicit
        // comparator: the map's own sort(true) orders DESCENDING by score, worst first.
        recoToTruth->sort(byAscendingScore);
        // Every declared instance label must be put on every path, including the one
        // where the reco collection was absent: a missing put is a framework error.
        event.put(std::move(recoToTruth), key + "RecoToTruth" + wp.name);
      }
    } else {
      // One map per working point, filled together: the candidate list is the whole
      // per-object cost and every working point only re-ranks it, so it is computed
      // once per object rather than once per (object, working point).
      std::vector<std::unique_ptr<MapType>> recoToTruthPerWp;
      recoToTruthPerWp.reserve(workingPoints_.size());
      for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
        recoToTruthPerWp.push_back(std::make_unique<MapType>(nReco));
      }

      // The candidates of one reco object, restricted to the roots an adaptive point may
      // answer with. Declared here and cleared per object, so it allocates once.
      std::vector<truth::BranchMatch> assignableMatches;

      for (unsigned int i = 0; i < nReco; ++i) {
        if (recoHitsPerObject[i].empty()) {
          continue;
        }
        const std::span<const truth::RecoHit> span(recoHitsPerObject[i]);
        const auto matches = hitAssociator->bestBranches(span);

        // A candidate root carries the hits of its whole subgraph, so a parton, a beam
        // particle or an invented node covers the reco object entirely and wins on score.
        // An adaptive point may not answer with one. The barred roots stay in the first
        // working point's map, which the truth-driven direction reads its pair scores
        // from, so a level made of those particles keeps its efficiency.
        assignableMatches.clear();
        if (anyAdaptiveWorkingPoint) {
          for (auto const& match : matches) {
            if (match.rootParticleId < isAssignable.size() && isAssignable[match.rootParticleId] != 0) {
              assignableMatches.push_back(match);
            }
          }
        }
        // Ascending score, tightest first, as bestBranches ordered it: the filter keeps
        // the order, so the climb still starts from the best candidate.
        const std::span<const truth::BranchMatch> assignableSpan(assignableMatches);

        // RECO to TRUTH: the working point drives the search, and the score is
        // reco-normalised, so 1 - score is the RECO purity.
        for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
          auto const& wp = workingPoints_[wpIndex];
          if (wp.adaptive) {
            const auto match =
                truth::BranchHitAssociator::bestAdaptiveBranch(assignableSpan, wp.reverseWeight, wp.maxReverseScore);
            if (match.rootParticleId != truth::BranchMatch::kInvalidRoot) {
              recoToTruthPerWp[wpIndex]->insert(i, match.rootParticleId, match.sharedEnergy, match.score);
            }
          } else {
            for (auto const& match : matches) {
              recoToTruthPerWp[wpIndex]->insert(i, match.rootParticleId, match.sharedEnergy, match.score);
            }
          }
        }

        // TRUTH to RECO, filled once per object. NO adaptive climb: the climb chooses
        // a graph level to suit the reco object, which is meaningless when the truth
        // target is the thing being asked about. Both payloads of this direction are
        // TRUTH-normalised: the sim-normalised shared energy fraction, which is the
        // axis HGCalValidator gates efficiency on, and the truth-normalised score,
        // which gates purity and duplicate. A shared-hits domain has no energy, so it
        // keeps reporting the shared hit count.
        constexpr bool sharedEnergyMetric = Traits::metric == truth::BranchHitAssociator::Metric::SharedEnergy;
        for (auto const& match : matches) {
          // The row index comes from the hit index and the map is sized from the graph.
          // A candidate outside the graph means the two products were built from
          // different events, which is a configuration error, not a row to write.
          if (match.rootParticleId >= nTruthRows) {
            std::call_once(rowOutOfRangeWarned_, [&key] {
              edm::LogWarning("AllRecoToTruthBranchAssociators")
                  << "Hit index and truth graph disagree on the particle count for '" << key
                  << "'; the candidates outside the graph are dropped. Check that both products come from the same "
                     "event.";
            });
            continue;
          }
          const float truthValue = sharedEnergyMetric ? match.sharedEnergyFraction : match.sharedEnergy;
          truthToReco->insert(match.rootParticleId, i, truthValue, match.reverseScore);
        }
      }

      for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
        // The rows keep the associator's order: ascending score, equal scores by the
        // tightest branch first, so [0] is the best match. A sort by score and index
        // here would hand a tie to the lowest id, an ancestor.
        // Every declared instance label must be put on every path, including the one
        // where the reco collection was absent: a missing put is a framework error.
        event.put(std::move(recoToTruthPerWp[wpIndex]), key + "RecoToTruth" + workingPoints_[wpIndex].name);
      }
    }
    truthToReco->sort(byAscendingScore);
    event.put(std::move(truthToReco), key + "TruthToReco");
  }
}

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
void AllRecoToTruthBranchAssociatorsProducer<RECO>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<std::vector<edm::InputTag>>("recoCollections", {});
  desc.add<edm::InputTag>("assignableTargetsSrc", edm::InputTag("truthBranchTargets", "assignableRoots"))
      ->setComment(
          "The roots an adaptive working point may answer with. The barred ones stay in targetsSrc and in the "
          "first working point's map, because they are the members of the hard-process and parton-jet "
          "denominators and the truth-driven direction reads its pair scores from that map");
  desc.add<edm::InputTag>("targetsSrc", edm::InputTag("truthBranchTargets", "selectedRoots"))
      ->setComment(
          "The selector-passing candidate roots from TruthBranchTargetsProducer, which also emits the level "
          "denominators and the signal seeds every associator module shares");
  desc.add<std::vector<std::string>>("denominatorDetectors", {})
      ->setComment(
          "DetId::Detector names the sim-normalised shared-energy denominator covers. Empty means the whole hit "
          "channel. One channel spans several detectors: HitChannel::Calo carries the barrel ECAL and HCAL "
          "deposits next to the HGCAL ones, and PCaloHit energies are sampling energies, so a branch that "
          "showered in the barrel has a channel-wide energy no endcap trackster can cover half of. Measured on "
          "200 no-PU ttbar events: 0.5% to 10% of a top branch's channel energy is in HGCAL, so the fraction was "
          "zero for every top");
  desc.add<bool>("truthToRecoSignalOnly", true)
      ->setComment(
          "Restrict the TruthToReco denominator to the signal interaction. Efficiency, duplicate and split are "
          "meaningless averaged over the overlaid pileup interactions. The associator's candidate set is NOT "
          "restricted, so pileup branches stay matchable and a pileup-matched reco object is not counted a fake");
  desc.add<double>("maxConstituentScore", 0.25)
      ->setComment(
          "Composite domains only. A constituent whose best match scores worse than this places the constituent "
          "at no truth vertex, because it would otherwise donate its whole pt^2 to a vertex it barely belongs to. "
          "1 - score is the constituent's reco purity, so the default is the 0.75 shared fraction the tracker "
          "association applies as Cut_RecoToSim, above which a reco-to-sim pair enters its map at all "
          "(SimTracker/TrackAssociatorProducers/python/quickTrackAssociatorByHits_cfi.py). The denominator stays "
          "over ALL constituents, as the reference vertex association does: an unmatched track legitimately "
          "lowers the shared fraction");
  desc.add<bool>("heavyFlavorOnly", false)
      ->setComment(
          "Composite domains only. Keep in the denominator only the vertices where a b or c hadron DECAYED, which "
          "is what inclusiveSecondaryVertices reconstructs: 4 and 5 per no-PU ttbar event against 4.1 "
          "reconstructed. Off by default; the secondary-vertex associator turns it on. Without it the denominator "
          "is every graph vertex with two selected roots, 45.9 per event, and the efficiency is capped by the "
          "denominator.");
  desc.add<std::vector<std::string>>("workingPointNames", {"Fixed"})
      ->setComment(
          "One RecoToTruth product per name. The name \"Fixed\" selects the plain per-root match; any other name "
          "makes the point adaptive, driven by its adaptiveReverseWeight and adaptiveMaxReverseScore entries. The "
          "FIRST listed point is the reference: the single TruthToReco product is computed at it, so list Fixed "
          "first");
  desc.add<std::vector<float>>("adaptiveReverseWeight", {0.f});
  desc.add<std::vector<float>>("adaptiveMaxReverseScore", {0.f});
  if constexpr (LayerClusterBackedRecoHits<RECO>) {
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<std::vector<edm::InputTag>>("hgcalRecHits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")})
        ->setComment("HGCAL rechits whose energies weight the cells of the shared-energy metric");
    desc.add<std::vector<edm::InputTag>>(
            "pfRecHits", {edm::InputTag("particleFlowRecHitECAL"), edm::InputTag("particleFlowRecHitHBHE")})
        ->setComment(
            "Barrel PFRecHits whose energies weight the cells of the shared-energy metric; the collections the "
            "barrel layer clusters are built from, not the Cleaned instances");
  }
  if constexpr (ConstituentBasedDomain<RECO>) {
    desc.add<std::string>("constituentAssociator", "allTrackToTruthBranchAssociators")
        ->setComment(
            "Module that produced the constituents' association maps. Its map rows must be sorted ascending by "
            "score, best first, as this package's producers emit them; the constituent scan reads row [0] as the "
            "best match and stops at the first row above maxConstituentScore");
    desc.add<std::string>("constituentCollection", "generalTracks")
        ->setComment("Constituent collection key, used to rebuild the instance labels");
    desc.add<std::string>("vertexResolution", "immediate")
        ->setComment(
            "Which truth vertex a constituent counts at: 'immediate' is the production vertex of its matched "
            "particle, right for a secondary vertex; 'interaction' is the production vertex of that particle's "
            "topmost ancestor, right for a primary vertex, where a track from a downstream decay still belongs "
            "to the interaction the chain started from");
  }
  descriptions.add(Traits::cfiName, desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using AllTrackToTruthBranchAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<reco::Track>;
DEFINE_FWK_MODULE(AllTrackToTruthBranchAssociatorsProducer);
using AllVertexToTruthBranchAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<reco::Vertex>;
DEFINE_FWK_MODULE(AllVertexToTruthBranchAssociatorsProducer);
// NOT named AllTracksterToTruthBranchAssociatorsProducer: a standalone producer of that
// name exists on the NanoAOD training branch, with different product keying and a
// different candidate-root source. A duplicate class name would make the framework pick
// one of the two at random in an area that carries both.
using TruthBranchTracksterAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<ticl::Trackster>;
DEFINE_FWK_MODULE(TruthBranchTracksterAssociatorsProducer);
using TruthBranchPFClusterAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<reco::PFCluster>;
DEFINE_FWK_MODULE(TruthBranchPFClusterAssociatorsProducer);
