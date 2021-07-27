#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

class TrackProducerFromPatMuons : public edm::stream::EDProducer<> {
public:
  explicit TrackProducerFromPatMuons(const edm::ParameterSet &);
  ~TrackProducerFromPatMuons() {}

  //   static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  virtual void produce(edm::Event &, const edm::EventSetup &) override;

  edm::EDGetTokenT<std::vector<pat::Muon>> inputMuons_;
  edm::EDPutTokenT<reco::TrackCollection> outputTrack_;
  bool innerTrackOnly_;
};

TrackProducerFromPatMuons::TrackProducerFromPatMuons(const edm::ParameterSet &iConfig)

{
  inputMuons_ = consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("src"));
  outputTrack_ = produces<reco::TrackCollection>();
  innerTrackOnly_ = iConfig.getParameter<bool>("innerTrackOnly");
}

// ------------ method called for each event  ------------
void TrackProducerFromPatMuons::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  Handle<std::vector<pat::Muon>> muons;
  iEvent.getByToken(inputMuons_, muons);

  reco::TrackCollection tracksOut;

  for (auto const &muon : *muons) {
    const reco::TrackRef trackRef = innerTrackOnly_ ? muon.innerTrack() : muon.muonBestTrack();
    if (trackRef.isNonnull() && trackRef->extra().isAvailable()) {
      tracksOut.emplace_back(*trackRef);
    }
  }

  iEvent.emplace(outputTrack_, std::move(tracksOut));
}

DEFINE_FWK_MODULE(TrackProducerFromPatMuons);
