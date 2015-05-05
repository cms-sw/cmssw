#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleNumberOfLayers.h"

/**
 * The purpose of this producer is to create a ValueMap<pair<unsigned int, unsigned int>>
 * for the number of pixel and strip stereo tracker layers the TrackingParticle has hits on.
 */
class TrackingParticleNumberOfLayersProducer: public edm::global::EDProducer<> {
public:
  TrackingParticleNumberOfLayersProducer(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  std::vector<edm::EDGetTokenT<std::vector<PSimHit>>> simHitTokens_;
};


TrackingParticleNumberOfLayersProducer::TrackingParticleNumberOfLayersProducer(const edm::ParameterSet& iConfig):
  tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles")))
{
  for(const auto& tag: iConfig.getParameter<std::vector<edm::InputTag>>("simHits")) {
    simHitTokens_.push_back(consumes<std::vector<PSimHit>>(tag));
  }

  produces<edm::ValueMap<unsigned int>>("trackerLayers");
  produces<edm::ValueMap<unsigned int>>("pixelLayers");
  produces<edm::ValueMap<unsigned int>>("stripStereoLayers");
}

void TrackingParticleNumberOfLayersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));

  desc.add<std::vector<edm::InputTag> >("simHits", {
      edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
      edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
      edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof"),
      edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
      edm::InputTag("g4SimHits", "TrackerHitsTECHighTof"),
      edm::InputTag("g4SimHits", "TrackerHitsTECLowTof"),
      edm::InputTag("g4SimHits", "TrackerHitsTIBHighTof"),
      edm::InputTag("g4SimHits", "TrackerHitsTIBLowTof"),
      edm::InputTag("g4SimHits", "TrackerHitsTIDHighTof"),
      edm::InputTag("g4SimHits", "TrackerHitsTIDLowTof"),
      edm::InputTag("g4SimHits", "TrackerHitsTOBHighTof"),
      edm::InputTag("g4SimHits", "TrackerHitsTOBLowTof")
    });

  descriptions.add("trackingParticleNumberOfLayersProducer", desc);
}

void TrackingParticleNumberOfLayersProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<TrackingParticleCollection> htps;
  iEvent.getByToken(tpToken_, htps);

  TrackingParticleNumberOfLayers algo(iEvent, simHitTokens_);
  auto ret = algo.calculate(htps, iSetup);
  iEvent.put(std::move(std::get<TrackingParticleNumberOfLayers::nTrackerLayers>(ret)), "trackerLayers");
  iEvent.put(std::move(std::get<TrackingParticleNumberOfLayers::nPixelLayers>(ret)), "pixelLayers");
  iEvent.put(std::move(std::get<TrackingParticleNumberOfLayers::nStripMonoAndStereoLayers>(ret)), "stripStereoLayers");
}

DEFINE_FWK_MODULE(TrackingParticleNumberOfLayersProducer);
