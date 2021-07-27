#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

class TrackExtraRekeyer : public edm::stream::EDProducer<>
{
public:
  explicit TrackExtraRekeyer(const edm::ParameterSet &);
  ~TrackExtraRekeyer() {}

//   static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  
  virtual void produce(edm::Event &, const edm::EventSetup &) override;
  
  edm::EDGetTokenT<reco::TrackCollection> inputTrack_;
  edm::EDGetTokenT<edm::Association<reco::TrackExtraCollection>> inputAssoc_;
  edm::EDPutTokenT<reco::TrackCollection> outputTrack_;

};


TrackExtraRekeyer::TrackExtraRekeyer(const edm::ParameterSet &iConfig)

{
  inputTrack_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"));
  inputAssoc_ = consumes<edm::Association<reco::TrackExtraCollection>>(iConfig.getParameter<edm::InputTag>("association"));
  
  outputTrack_ = produces<reco::TrackCollection>();
}

// ------------ method called for each event  ------------
void TrackExtraRekeyer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  using namespace edm;
  
  Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(inputTrack_, tracks);
  
  Handle<edm::Association<reco::TrackExtraCollection>> assoc;
  iEvent.getByToken(inputAssoc_, assoc);

  reco::TrackCollection tracksOut;
  
  for (auto const &track : *tracks) {
    if (!assoc->contains(track.extra().id())) {
      continue;
    }
    const reco::TrackExtraRef &trackextraref = (*assoc)[track.extra()];
    if (trackextraref.isNonnull()) {
      auto &trackout = tracksOut.emplace_back(track);
      trackout.setExtra(trackextraref);
    }
  }
  
  iEvent.emplace(outputTrack_, std::move(tracksOut));
  
}

DEFINE_FWK_MODULE(TrackExtraRekeyer);
