#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

class TrackProducerFromPatMuons : public edm::global::EDProducer<> {
public:
  explicit TrackProducerFromPatMuons(const edm::ParameterSet &);
  ~TrackProducerFromPatMuons() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  const edm::EDGetTokenT<std::vector<pat::Muon>> inputMuons_;
  const edm::EDPutTokenT<reco::TrackCollection> outputTrack_;
  const bool innerTrackOnly_;
};

TrackProducerFromPatMuons::TrackProducerFromPatMuons(const edm::ParameterSet &iConfig)
    : inputMuons_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("src"))),
      outputTrack_(produces<reco::TrackCollection>()),
      innerTrackOnly_(iConfig.getParameter<bool>("innerTrackOnly")) {}

// ------------ method called for each event  ------------
void TrackProducerFromPatMuons::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  auto const &muons = iEvent.get(inputMuons_);

  reco::TrackCollection tracksOut;

  for (auto const &muon : muons) {
    const reco::TrackRef trackRef = innerTrackOnly_ ? muon.innerTrack() : muon.muonBestTrack();
    if (trackRef.isNonnull() && trackRef->extra().isAvailable()) {
      tracksOut.emplace_back(*trackRef);
    }
  }
  iEvent.emplace(outputTrack_, std::move(tracksOut));
}

void TrackProducerFromPatMuons::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Simple prooducer to generate track from pat::muons ");
  desc.add<edm::InputTag>("src", edm::InputTag("slimmedMuons"))->setComment("input track collections");
  desc.add<bool>("innerTrackOnly", true)->setComment("use only inner track");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TrackProducerFromPatMuons);
