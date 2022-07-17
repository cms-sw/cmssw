//
//

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <vector>
#include <memory>
#include <cmath>

class MuonSelectorVertex : public edm::global::EDProducer<> {
public:
  explicit MuonSelectorVertex(const edm::ParameterSet& iConfig);
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  const edm::EDGetTokenT<std::vector<pat::Muon>> muonSource_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> vertexSource_;
  const double maxDZ_;
};

MuonSelectorVertex::MuonSelectorVertex(const edm::ParameterSet& iConfig)
    : muonSource_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muonSource"))),
      vertexSource_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertexSource"))),
      maxDZ_(iConfig.getParameter<double>("maxDZ")) {
  produces<std::vector<pat::Muon>>();
}

void MuonSelectorVertex::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& muons = iEvent.get(muonSource_);
  const auto& vertices = iEvent.get(vertexSource_);

  auto selectedMuons = std::make_unique<std::vector<pat::Muon>>();

  if (!vertices.empty()) {
    for (const auto& muon : muons) {
      if (std::abs(muon.vertex().z() - vertices.at(0).z()) < maxDZ_) {
        selectedMuons->push_back(muon);
      }
    }
  }

  iEvent.put(std::move(selectedMuons));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonSelectorVertex);
