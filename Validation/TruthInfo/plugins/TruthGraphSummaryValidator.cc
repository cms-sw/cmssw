// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// What the truth graph of this event contains, per interaction: how many interactions it
// holds, how big each one is, which artificial vertices it carries and how many members
// each level has, signal and pile-up apart. These are the numbers that say whether the
// graph was built as intended; without them a pile-up regression is only visible by
// dumping a graph by hand.

#include <string>
#include <unordered_map>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace {

  struct Histograms {
    dqm::reco::MonitorElement* interactions = nullptr;
    dqm::reco::MonitorElement* particlesPerInteraction = nullptr;
    dqm::reco::MonitorElement* verticesPerInteraction = nullptr;
    dqm::reco::MonitorElement* signalParticles = nullptr;
    dqm::reco::MonitorElement* pileupParticles = nullptr;
    dqm::reco::MonitorElement* particlesWithoutMomentum = nullptr;
    dqm::reco::MonitorElement* vertexRoles = nullptr;
    dqm::reco::MonitorElement* levelMembersSignal = nullptr;
    dqm::reco::MonitorElement* levelMembersPileup = nullptr;
  };

  constexpr int kNVertexRoles = truth::kVertexRoleCount;

  // One bin per level plus the signal flag, the order levelNamesOf reports.
  constexpr int kNLevelBins = static_cast<int>(truth::kLevelTable.size()) + 1;

}  // namespace

class TruthGraphSummaryValidator : public DQMGlobalEDAnalyzer<Histograms> {
public:
  explicit TruthGraphSummaryValidator(edm::ParameterSet const& pset)
      : token_(consumes<truth::Graph>(pset.getParameter<edm::InputTag>("src"))),
        folder_(pset.getParameter<std::string>("folder")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<std::string>("folder", "TruthInfo/Graph");
    descriptions.addWithDefaultLabel(desc);
  }

  void bookHistograms(dqm::reco::DQMStore::IBooker& booker,
                      edm::Run const&,
                      edm::EventSetup const&,
                      Histograms& histograms) const override {
    booker.setCurrentFolder(folder_);
    histograms.interactions = booker.book1D("interactions", "interactions per event", 512, -0.5, 511.5);
    histograms.particlesPerInteraction =
        booker.book1D("particles_per_interaction", "particles per interaction", 200, 0., 10000.);
    histograms.verticesPerInteraction =
        booker.book1D("vertices_per_interaction", "vertices per interaction", 200, 0., 10000.);
    histograms.signalParticles = booker.book1D("signal_particles", "signal particles per event", 200, 0., 20000.);
    histograms.pileupParticles = booker.book1D("pileup_particles", "pileup particles per event", 200, 0., 400000.);
    // A GEN particle of a pile-up interaction has a momentum only through its SimTrack or
    // its decay products, so this counts what no kinematic cut can select.
    histograms.particlesWithoutMomentum =
        booker.book1D("particles_without_momentum", "particles with no momentum per event", 200, 0., 100000.);

    histograms.vertexRoles =
        booker.book1D("vertex_roles", "vertices per role per event", kNVertexRoles, -0.5, kNVertexRoles - 0.5);
    for (int role = 0; role < kNVertexRoles; ++role) {
      histograms.vertexRoles->setBinLabel(role + 1, truth::vertexRoleName(static_cast<truth::VertexRole>(role)));
    }

    auto bookLevels = [&booker](std::string const& name, std::string const& title) {
      auto* me = booker.book1D(name, title, kNLevelBins, -0.5, kNLevelBins - 0.5);
      int bin = 1;
      for (auto const& row : truth::kLevelTable) {
        me->setBinLabel(bin++, row.name);
      }
      me->setBinLabel(bin, truth::kSignalLevelName);
      return me;
    };
    histograms.levelMembersSignal = bookLevels("level_members_signal", "signal level members per event");
    histograms.levelMembersPileup = bookLevels("level_members_pileup", "pileup level members per event");
  }

  void dqmAnalyze(edm::Event const& event, edm::EventSetup const&, Histograms const& histograms) const override {
    auto const& graph = event.get(token_);

    std::unordered_map<uint64_t, std::size_t> interactions;
    std::vector<uint32_t> particlesOf;
    std::size_t signalParticles = 0;
    std::size_t withoutMomentum = 0;

    auto indexOfInteraction = [&interactions](uint64_t eventId) {
      return interactions.try_emplace(eventId, interactions.size()).first->second;
    };

    for (auto const& particle : graph.particles()) {
      const std::size_t interaction = indexOfInteraction(particle.eventId);
      if (interaction >= particlesOf.size()) {
        particlesOf.resize(interaction + 1, 0);
      }
      ++particlesOf[interaction];

      if (particle.isSignal()) {
        ++signalParticles;
      }
      if (!particle.hasMomentum()) {
        ++withoutMomentum;
      }

      auto* levels = particle.isSignal() ? histograms.levelMembersSignal : histograms.levelMembersPileup;
      int bin = 0;
      for (auto const& row : truth::kLevelTable) {
        if (particle.isAtLevel(row.flag)) {
          levels->Fill(bin);
        }
        ++bin;
      }
      if (particle.isAtLevel(truth::LevelFlag::Signal)) {
        levels->Fill(bin);
      }
    }

    std::vector<uint32_t> verticesOf(interactions.size(), 0);
    for (auto const& vertex : graph.vertices()) {
      const std::size_t interaction = indexOfInteraction(vertex.eventId);
      if (interaction >= verticesOf.size()) {
        verticesOf.resize(interaction + 1, 0);
      }
      ++verticesOf[interaction];
      histograms.vertexRoles->Fill(static_cast<int>(vertex.role));
    }

    histograms.interactions->Fill(interactions.size());
    histograms.signalParticles->Fill(signalParticles);
    histograms.pileupParticles->Fill(graph.nParticles() - signalParticles);
    histograms.particlesWithoutMomentum->Fill(withoutMomentum);
    for (std::size_t i = 0; i < interactions.size(); ++i) {
      histograms.particlesPerInteraction->Fill(i < particlesOf.size() ? particlesOf[i] : 0);
      histograms.verticesPerInteraction->Fill(i < verticesOf.size() ? verticesOf[i] : 0);
    }
  }

private:
  const edm::EDGetTokenT<truth::Graph> token_;
  const std::string folder_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthGraphSummaryValidator);
