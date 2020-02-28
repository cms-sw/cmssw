////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class GsfElectronFromPVSelector : public edm::global::EDProducer<> {
public:
  explicit GsfElectronFromPVSelector(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  double max_dxy_;
  double max_dz_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> v_recoVertexToken_;
  edm::EDGetTokenT<std::vector<reco::GsfElectron>> v_recoGsfElectronToken_;
};

////////////////////////////////////////////////////////////////////////////////
// construction
////////////////////////////////////////////////////////////////////////////////

GsfElectronFromPVSelector::GsfElectronFromPVSelector(edm::ParameterSet const& iConfig)
    : max_dxy_{iConfig.getParameter<double>("max_dxy")},
      max_dz_{iConfig.getParameter<double>("max_dz")},
      v_recoVertexToken_{consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("srcVertex"))},
      v_recoGsfElectronToken_{
          consumes<std::vector<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("srcElectron"))} {
  produces<std::vector<reco::GsfElectron>>();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

void GsfElectronFromPVSelector::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<std::vector<reco::Vertex>> vertices;
  iEvent.getByToken(v_recoVertexToken_, vertices);

  edm::Handle<std::vector<reco::GsfElectron>> gsfElectrons;
  iEvent.getByToken(v_recoGsfElectronToken_, gsfElectrons);

  auto goodGsfElectrons = std::make_unique<std::vector<reco::GsfElectron>>();

  if (!vertices->empty() && !gsfElectrons->empty()) {
    auto const& pv = vertices->front();
    std::copy_if(std::cbegin(*gsfElectrons),
                 std::cend(*gsfElectrons),
                 std::back_inserter(*goodGsfElectrons),
                 [this, &pv](auto const& GsfElectron) {
                   return std::abs(GsfElectron.gsfTrack()->dxy(pv.position())) < max_dxy_ &&
                          std::abs(GsfElectron.gsfTrack()->dz(pv.position())) < max_dz_;
                 });
  }

  iEvent.put(std::move(goodGsfElectrons));
}

DEFINE_FWK_MODULE(GsfElectronFromPVSelector);
