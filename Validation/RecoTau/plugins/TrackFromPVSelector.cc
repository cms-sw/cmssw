////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// -- new -- [GS]
//#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class TrackFromPVSelector : public edm::global::EDProducer<> {
public:
  explicit TrackFromPVSelector(edm::ParameterSet const& iConfig);

  void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

private:
  double max_dxy_;
  double max_dz_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> v_recoVertexToken_;
  //edm::EDGetTokenT<std::vector<reco::Track>> v_recoTrackToken_;
  edm::EDGetTokenT<std::vector<pat::PackedCandidate>> v_recoTrackToken_;
};

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TrackFromPVSelector::TrackFromPVSelector(edm::ParameterSet const& iConfig)
    : max_dxy_{iConfig.getParameter<double>("max_dxy")},
      max_dz_{iConfig.getParameter<double>("max_dz")},
      v_recoVertexToken_{consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("srcVertex"))},
      //v_recoTrackToken_{consumes<std::vector<reco::Track>>(iConfig.getParameter<edm::InputTag>("srcTrack"))} {
      v_recoTrackToken_{consumes<std::vector<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("srcTrack"))} {
  //produces<std::vector<reco::Track>>();
  produces<std::vector<pat::PackedCandidate>>();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void TrackFromPVSelector::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<std::vector<reco::Vertex>> vertices;
  iEvent.getByToken(v_recoVertexToken_, vertices);

  //edm::Handle<std::vector<reco::Track>> tracks;
  edm::Handle<std::vector<pat::PackedCandidate>> tracks;
  iEvent.getByToken(v_recoTrackToken_, tracks);

  //auto goodTracks = std::make_unique<std::vector<reco::Track>>();
  auto goodTracks = std::make_unique<std::vector<pat::PackedCandidate>>();
  if (!vertices->empty() && !tracks->empty()) {
    auto const& vtxPos = vertices->front().position();
    std::copy_if(
        std::cbegin(*tracks), std::cend(*tracks), std::back_inserter(*goodTracks), [this, &vtxPos](auto const& track) {
          return std::abs(track.dxy(vtxPos)) < max_dxy_ && std::abs(track.dz(vtxPos)) < max_dz_;
        });
  }
  iEvent.put(std::move(goodTracks));
}

DEFINE_FWK_MODULE(TrackFromPVSelector);
