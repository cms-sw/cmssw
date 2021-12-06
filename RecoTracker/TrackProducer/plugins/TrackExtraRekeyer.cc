#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

class TrackExtraRekeyer : public edm::global::EDProducer<> {
public:
  explicit TrackExtraRekeyer(const edm::ParameterSet &);
  ~TrackExtraRekeyer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  // memeber data
  const edm::EDGetTokenT<reco::TrackCollection> inputTrack_;
  const edm::EDGetTokenT<edm::Association<reco::TrackExtraCollection>> inputAssoc_;
  const edm::EDPutTokenT<reco::TrackCollection> outputTrack_;
};

TrackExtraRekeyer::TrackExtraRekeyer(const edm::ParameterSet &iConfig)
    : inputTrack_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      inputAssoc_(consumes(iConfig.getParameter<edm::InputTag>("association"))),
      outputTrack_(produces<reco::TrackCollection>()) {}

// ------------ method called for each event  ------------
void TrackExtraRekeyer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  auto const &tracks = iEvent.get(inputTrack_);
  auto const &assoc = iEvent.get(inputAssoc_);

  reco::TrackCollection tracksOut;

  for (auto const &track : tracks) {
    if (!assoc.contains(track.extra().id())) {
      continue;
    }
    const reco::TrackExtraRef &trackextraref = assoc[track.extra()];
    if (trackextraref.isNonnull()) {
      auto &trackout = tracksOut.emplace_back(track);
      trackout.setExtra(trackextraref);
    }
  }
  iEvent.emplace(outputTrack_, std::move(tracksOut));
}

void TrackExtraRekeyer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Simple prooducer to re-key muon tracks for refit");
  desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"))->setComment("input track collections");
  desc.add<edm::InputTag>("association", edm::InputTag("muonReducedTrackExtras"))
      ->setComment("input track association collection");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TrackExtraRekeyer);
