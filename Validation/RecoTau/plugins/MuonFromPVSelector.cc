#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <memory>
#include <numeric>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class MuonFromPVSelector : public edm::global::EDProducer<> {
public:
  explicit MuonFromPVSelector(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  double max_dxy_;
  double max_dz_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> v_recoVertexToken_;
  edm::EDGetTokenT<std::vector<reco::Muon>> v_recoMuonToken_;
};

////////////////////////////////////////////////////////////////////////////////
// construction
////////////////////////////////////////////////////////////////////////////////

MuonFromPVSelector::MuonFromPVSelector(edm::ParameterSet const& iConfig)
    : max_dxy_{iConfig.getParameter<double>("max_dxy")},
      max_dz_{iConfig.getParameter<double>("max_dz")},
      v_recoVertexToken_{consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("srcVertex"))},
      v_recoMuonToken_{consumes<std::vector<reco::Muon>>(iConfig.getParameter<edm::InputTag>("srcMuon"))} {
  produces<std::vector<reco::Muon>>();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

void MuonFromPVSelector::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto goodMuons = std::make_unique<std::vector<reco::Muon>>();

  edm::Handle<std::vector<reco::Vertex>> vertices;
  iEvent.getByToken(v_recoVertexToken_, vertices);

  edm::Handle<std::vector<reco::Muon>> muons;
  iEvent.getByToken(v_recoMuonToken_, muons);

  if (!vertices->empty()) {
    auto const& pv = vertices->front();
    std::copy_if(std::cbegin(*muons), std::cend(*muons), std::back_inserter(*goodMuons), [&pv, this](auto const& muon) {
      return muon.innerTrack().isNonnull() && std::abs(muon.innerTrack()->dxy(pv.position())) < max_dxy_ &&
             std::abs(muon.innerTrack()->dz(pv.position())) < max_dz_;
    });
  }

  iEvent.put(std::move(goodMuons));
}

DEFINE_FWK_MODULE(MuonFromPVSelector);
