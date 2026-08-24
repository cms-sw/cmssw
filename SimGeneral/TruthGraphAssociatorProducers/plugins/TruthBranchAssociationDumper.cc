// Counts what is actually IN the truth-branch association maps.
//
// A product being present proves only that the producer ran and put something. This
// reports rows, non-empty rows and total entries per map, which is what distinguishes
// a working associator from one that quietly wrote empty maps on every event.

#include <limits>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

namespace {
  using SharedHitsMap = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  using FractionMap = ticl::TICLAssociationMap<ticl::mapWithFractionAndScore>;

  template <typename MAP>
  void report(edm::Event const& event, std::vector<std::pair<std::string, edm::EDGetTokenT<MAP>>> const& tokens) {
    for (auto const& [name, token] : tokens) {
      edm::Handle<MAP> handle;
      event.getByToken(token, handle);
      if (!handle.isValid()) {
        edm::LogPrint("TruthAssoc") << "  " << name << ": PRODUCT NOT FOUND";
        continue;
      }
      auto const& map = handle->getMap();
      std::size_t nonEmpty = 0;
      std::size_t entries = 0;
      float bestScore = std::numeric_limits<float>::infinity();
      for (auto const& row : map) {
        if (!row.empty()) {
          ++nonEmpty;
          entries += row.size();
          bestScore = std::min(bestScore, row[0].score());
        }
      }
      edm::LogPrint("TruthAssoc") << "  " << name << ": rows=" << map.size() << " nonEmpty=" << nonEmpty
                                  << " entries=" << entries
                                  << (nonEmpty > 0 ? "  bestScore=" + std::to_string(bestScore) : "");
    }
  }
}  // namespace

class TruthBranchAssociationDumper : public edm::one::EDAnalyzer<> {
public:
  explicit TruthBranchAssociationDumper(edm::ParameterSet const& cfg) {
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("sharedHitsMaps")) {
      hitsTokens_.emplace_back(tag.encode(), consumes<SharedHitsMap>(tag));
    }
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("fractionMaps")) {
      fractionTokens_.emplace_back(tag.encode(), consumes<FractionMap>(tag));
    }
    // Composite domains are only meaningful if their constituents exist: a primary
    // vertex with no tracks is correctly skipped, and that must not be mistaken for a
    // broken association.
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("vertexDiagnostics")) {
      vertexTokens_.emplace_back(tag.encode(), consumes<std::vector<reco::Vertex>>(tag));
    }
  }

  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    edm::LogPrint("TruthAssoc") << "=== event " << event.id().event() << " ===";
    report(event, hitsTokens_);
    report(event, fractionTokens_);
    for (auto const& [name, token] : vertexTokens_) {
      edm::Handle<std::vector<reco::Vertex>> handle;
      event.getByToken(token, handle);
      if (!handle.isValid()) {
        continue;
      }
      for (std::size_t i = 0; i < handle->size(); ++i) {
        auto const& v = (*handle)[i];
        edm::LogPrint("TruthAssoc") << "  " << name << "[" << i << "]: tracks=" << v.tracksSize()
                                    << " isFake=" << v.isFake() << " ndof=" << v.ndof();
      }
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<edm::InputTag>>("sharedHitsMaps", {});
    desc.add<std::vector<edm::InputTag>>("fractionMaps", {});
    desc.add<std::vector<edm::InputTag>>("vertexDiagnostics", {});
    descriptions.addWithDefaultLabel(desc);
  }

private:
  std::vector<std::pair<std::string, edm::EDGetTokenT<SharedHitsMap>>> hitsTokens_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<FractionMap>>> fractionTokens_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<reco::Vertex>>>> vertexTokens_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthBranchAssociationDumper);
