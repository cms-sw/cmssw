////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <vector>
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class bestPVselector : public edm::global::EDProducer<> {
public:
  explicit bestPVselector(edm::ParameterSet const& iConfig);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  edm::EDGetTokenT<std::vector<reco::Vertex>> src_;
};

////////////////////////////////////////////////////////////////////////////////
// construction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bestPVselector::bestPVselector(edm::ParameterSet const& iConfig)
    : src_{consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("src"))} {
  produces<std::vector<reco::Vertex>>();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void bestPVselector::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<std::vector<reco::Vertex>> vertices;
  iEvent.getByToken(src_, vertices);

  auto theBestPV = std::make_unique<std::vector<reco::Vertex>>();

  if (!vertices->empty()) {
    auto sumSquarePt = [](auto const& pv) { return pv.p4().pt() * pv.p4().pt(); };
    auto bestPV =
        std::max_element(std::cbegin(*vertices), std::cend(*vertices), [sumSquarePt](auto const& v1, auto const& v2) {
          return sumSquarePt(v1) < sumSquarePt(v2);
        });
    theBestPV->push_back(*bestPV);
  }
  iEvent.put(std::move(theBestPV));
}

using HighestSumP4PrimaryVertexSelector = bestPVselector;
DEFINE_FWK_MODULE(HighestSumP4PrimaryVertexSelector);
