// The truth-side target products every associator and validator consumes: the
// selector-passing candidate roots, the signal-seed denominators, and one TruthToReco
// denominator per graph level with its eligibility mask. They depend only on the graph
// and on the selection configuration, never on a reco collection. One producer computes
// them once per event, so every consumer sees the same targets and the same cuts.

#include <cctype>
#include <limits>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

class TruthBranchTargetsProducer : public edm::global::EDProducer<> {
public:
  explicit TruthBranchTargetsProducer(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  truth::BranchSelector branchSelector_;
  // (level, product instance) pairs, instance = "truthToRecoTargets" + capitalized name.
  std::vector<std::pair<truth::Level, std::string>> truthLevels_;
  const std::vector<int> signalSeedPdgIds_;
  const std::vector<int> signalSeedHadronFlavors_;
  const bool truthToRecoSignalOnly_;
};

TruthBranchTargetsProducer::TruthBranchTargetsProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      signalSeedPdgIds_(cfg.getParameter<std::vector<int>>("signalSeedPdgIds")),
      signalSeedHadronFlavors_(cfg.getParameter<std::vector<int>>("signalSeedHadronFlavors")),
      truthToRecoSignalOnly_(cfg.getParameter<bool>("truthToRecoSignalOnly")) {
  {
    // Restrict which branches are candidates at all. Without this the maps and the
    // efficiency denominators are dominated by soft particles that no reconstruction
    // was ever going to find, exactly as CaloParticleSelector and the TrackingParticle
    // selectors guard their own denominators.
    auto const& sel = cfg.getParameter<edm::ParameterSet>("branchSelector");
    truth::BranchSelector::Config selectorConfig;
    selectorConfig.ptMin = sel.getParameter<float>("ptMin");
    selectorConfig.ptMax = sel.getParameter<float>("ptMax");
    selectorConfig.etaMin = sel.getParameter<float>("etaMin");
    selectorConfig.etaMax = sel.getParameter<float>("etaMax");
    selectorConfig.pdgIds = sel.getParameter<std::vector<int>>("pdgIds");
    selectorConfig.signalOnly = sel.getParameter<bool>("signalOnly");
    selectorConfig.intimeOnly = sel.getParameter<bool>("intimeOnly");
    selectorConfig.chargedOnly = sel.getParameter<bool>("chargedOnly");
    selectorConfig.invertEta = sel.getParameter<bool>("invertEta");
    selectorConfig.kinematicsOnStableOnly = sel.getParameter<bool>("kinematicsOnStableOnly");
    branchSelector_ = truth::BranchSelector(std::move(selectorConfig));
  }

  // The associators' candidate roots. NOT an efficiency denominator: the set can hold a
  // particle together with its own ancestor, so it is not an antichain.
  produces<std::vector<unsigned int>>("selectedRoots");
  produces<std::vector<unsigned int>>("signalSeeds");
  // The same seed species without any selector cut, so an efficiency can be quoted
  // against EVERY seed in the event and not only against those the kinematic selection
  // kept. The two denominators together separate "not reconstructed" from "never
  // offered": on 200 no-PU ttbar events the selector keeps 390 of the 400 tops.
  produces<std::vector<unsigned int>>("signalSeedsNoSelection");
  // One denominator product per configured level, labelled
  // "truthToRecoTargets" + the level name with its first letter capitalized.
  for (auto const& name : cfg.getParameter<std::vector<std::string>>("truthLevels")) {
    if (name.empty()) {
      throw cms::Exception("Configuration") << "empty entry in truthLevels";
    }
    std::string capitalized = name;
    capitalized[0] = std::toupper(static_cast<unsigned char>(capitalized[0]));
    truthLevels_.emplace_back(truth::levelFromName(name), "truthToRecoTargets" + capitalized);
    produces<std::vector<unsigned int>>(truthLevels_.back().second);
    // Parallel to the denominator: which plotted-axis cut each target fails.
    produces<std::vector<unsigned int>>(truthLevels_.back().second + "Eligibility");
  }
}

void TruthBranchTargetsProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  const unsigned int nBranches = graph.nParticles();

  // Selected candidate roots. If the selection accepts nothing the answer is "no
  // candidates", not "every particle", which would silently undo the selection.
  auto selectedRoots = std::make_unique<std::vector<unsigned int>>();
  selectedRoots->reserve(nBranches);
  std::vector<bool> isCandidate(nBranches, false);
  for (uint32_t id = 0; id < nBranches; ++id) {
    if (branchSelector_(truth::Branch(&graph, id))) {
      selectedRoots->push_back(id);
      isCandidate[id] = true;
    }
  }

  // The preset seed objects: with a tau preset the tau roots alone, so the signal
  // efficiency is the tau's own, not its decay legs'.
  {
    auto const isSeedSpecies = [this, &graph](uint32_t id) {
      const int32_t pdgId = graph.particles()[id].pdgId;
      if (std::find(signalSeedPdgIds_.begin(), signalSeedPdgIds_.end(), pdgId) != signalSeedPdgIds_.end()) {
        return true;
      }
      for (const int flavor : signalSeedHadronFlavors_) {
        if (truth::hadronHasQuark(pdgId, flavor)) {
          return true;
        }
      }
      return false;
    };
    auto signalSeeds = std::make_unique<std::vector<unsigned int>>();
    auto signalSeedsNoSelection = std::make_unique<std::vector<unsigned int>>();
    // With no seed species there is no resonance in this sample, so BOTH products stay
    // EMPTY. Every selected root is not a substitute: that set holds particles together
    // with their own ancestors, so it is not an antichain and an efficiency over it
    // counts the same energy twice (on QCD it is 518.89 per event against 164
    // generator-stable particles).
    if (truth::seedsNameAResonance(signalSeedPdgIds_, signalSeedHadronFlavors_)) {
      for (uint32_t id : *selectedRoots) {
        if (isSeedSpecies(id)) {
          signalSeeds->push_back(id);
        }
      }
      for (uint32_t id = 0; id < nBranches; ++id) {
        if (isSeedSpecies(id)) {
          signalSeedsNoSelection->push_back(id);
        }
      }
    }
    event.put(std::move(signalSeeds), "signalSeeds");
    event.put(std::move(signalSeedsNoSelection), "signalSeedsNoSelection");
  }

  // eventId 0 is the signal interaction; anything else is overlaid pileup.
  auto isSignalParticle = [&graph](uint32_t particleId) { return graph.particles()[particleId].eventId == 0; };
  // One denominator per level. The level antichain, then the signal restriction, then
  // the kinematic selector. Order matters: taking the antichain of an already
  // kinematically-selected set would promote a soft particle to a level it does not
  // belong to just because its parent failed the pt cut.
  std::vector<unsigned int> extraCandidates;
  for (auto const& [level, instance] : truthLevels_) {
    auto targets = std::make_unique<std::vector<unsigned int>>();
    // Parallel to targets: which plotted-axis cut each one FAILS, 0 for those passing
    // both. An efficiency against pt must not have the pt cut applied to its own
    // denominator, so a target failing only the pt cut is kept and enters the pt plot
    // alone.
    auto eligibility = std::make_unique<std::vector<unsigned int>>();
    for (uint32_t id : truth::levelAntichain(graph, level)) {
      const truth::Branch branch(&graph, id);
      if (!branchSelector_.passesNonKinematic(branch)) {
        continue;
      }
      // No plot can suppress two cuts at once, so a branch failing more than one enters
      // none of them and is dropped here rather than carried and filtered everywhere.
      const uint32_t failed = branchSelector_.failedKinematicCuts(branch);
      if ((failed & (failed - 1u)) != 0u) {
        continue;
      }
      if (truthToRecoSignalOnly_ && !isSignalParticle(id)) {
        continue;
      }
      if (!isCandidate[id]) {
        isCandidate[id] = true;
        extraCandidates.push_back(id);
      }
      targets->push_back(id);
      eligibility->push_back(failed);
    }
    event.put(std::move(targets), instance);
    event.put(std::move(eligibility), instance + "Eligibility");
  }

  // Everything a denominator can ask about must be matchable, or its row is empty for
  // every reco collection and the plot that suppresses its own cut reads a structural
  // zero in the first bin. Exactly the targets emitted above join the candidates, and
  // not every particle that fails one cut: that would carry the soft tail of all 200
  // pileup interactions for no denominator, at 128% more time per PU200 ttbar event in
  // the track associator alone.
  selectedRoots->insert(selectedRoots->end(), extraCandidates.begin(), extraCandidates.end());
  std::sort(selectedRoots->begin(), selectedRoots->end());

  event.put(std::move(selectedRoots), "selectedRoots");
}

void TruthBranchTargetsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));

  edm::ParameterSetDescription selector;
  selector.add<float>("ptMin", 1.f)->setComment("Reject branches whose root is softer than this");
  selector.add<float>("ptMax", std::numeric_limits<float>::max());
  selector.add<float>("etaMin", -4.f);
  selector.add<float>("etaMax", 4.f);
  selector.add<std::vector<int>>("pdgIds", {})->setComment("Empty accepts every species");
  selector.add<bool>("signalOnly", false);
  selector.add<bool>("intimeOnly", false);
  selector.add<bool>("chargedOnly", false);
  selector.add<bool>("invertEta", false);
  selector.add<bool>("kinematicsOnStableOnly", true)
      ->setComment(
          "Apply ptMin/ptMax/etaMin/etaMax only to a root that decayed nowhere. The momentum of a root "
          "that decayed is not a detector observable: a resonance at rest has pt about 0 and |eta| "
          "unbounded, so a track-shaped cut rejects it while its decay products fill the calorimeter.");
  desc.add<edm::ParameterSetDescription>("branchSelector", selector);

  desc.add<std::vector<std::string>>("truthLevels", {"caloBoundary"})
      ->setComment("Graph levels to emit a TruthToReco denominator for, one product per level");
  desc.add<std::vector<int>>("signalSeedPdgIds", {})
      ->setComment("The selection preset's seed species; empty or {0} means no resonance and empty signal products");
  desc.add<std::vector<int>>("signalSeedHadronFlavors", {})
      ->setComment("Heavy-flavour hadron seeds; flavours alone also name a resonance");
  desc.add<bool>("truthToRecoSignalOnly", true)
      ->setComment("Restrict the level denominators to the signal interaction");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthBranchTargetsProducer);
