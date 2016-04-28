#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

class HitPairEDProducer: public edm::stream::EDProducer<> {
public:
  HitPairEDProducer(const edm::ParameterSet& iConfig);
  ~HitPairEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> seedingLayerToken_;
  edm::EDGetTokenT<edm::OwnVector<TrackingRegion> > regionToken_;

  edm::RunningAverage localRA_;
  LayerHitMapCache layerCache_;
  const unsigned int maxElement_;

  HitPairGeneratorFromLayerPair generator_;

  const bool produceSeedingHitSets_;
  const bool produceIntermediateHitDoublets_;
};


HitPairEDProducer::HitPairEDProducer(const edm::ParameterSet& iConfig):
  seedingLayerToken_(consumes<SeedingLayerSetsHits>(iConfig.getParameter<edm::InputTag>("seedingLayers"))),
  regionToken_(consumes<edm::OwnVector<TrackingRegion> >(iConfig.getParameter<edm::InputTag>("trackingRegions"))),
  maxElement_(iConfig.getParameter<unsigned int>("maxElement")),
  generator_(0, 1, nullptr, maxElement_), // TODO: make layer indices configurable?
  produceSeedingHitSets_(iConfig.getParameter<bool>("produceSeedingHitSets")),
  produceIntermediateHitDoublets_(iConfig.getParameter<bool>("produceIntermediateHitDoublets"))
{
  if(!produceIntermediateHitDoublets_ && !produceSeedingHitSets_)
    throw cms::Exception("Configuration") << "HitPairEDProducer requires either produceIntermediateHitDoublets or produceSeedingHitSets to be True. If neither are needed, just remove this module from your sequence/path as it doesn't do anything useful";

  if(produceSeedingHitSets_)
    produces<std::vector<SeedingHitSet> >();
  if(produceIntermediateHitDoublets_)
    produces<IntermediateHitDoublets>();
}

void HitPairEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("seedingLayers", edm::InputTag("seedingLayersEDProducer"));
  desc.add<edm::InputTag>("trackingRegions", edm::InputTag("globalTrackingRegionFromBeamSpot"));
  desc.add<bool>("produceSeedingHitSets", false);
  desc.add<bool>("produceIntermediateHitDoublets", false);
  desc.add<unsigned int>("maxElement", 0); // default is really 0? Also when used from CombinedHitTripletGenerator?

  descriptions.add("hitPairEDProducer", desc);
}

void HitPairEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<SeedingLayerSetsHits> hlayers;
  iEvent.getByToken(seedingLayerToken_, hlayers);
  const auto& layers = *hlayers;
  if(layers.numberOfLayersInSet() < 2)
    throw cms::Exception("Configuration") << "HitPairEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 2, got " << layers.numberOfLayersInSet();

  edm::Handle<edm::OwnVector<TrackingRegion> > hregions;
  iEvent.getByToken(regionToken_, hregions);
  const auto& regions = *hregions;

  std::unique_ptr<std::vector<SeedingHitSet> > seedingHitSets;
  if(produceSeedingHitSets_) {
    seedingHitSets = std::make_unique<std::vector<SeedingHitSet> >();
    seedingHitSets->reserve(localRA_.upper());
  }
  std::unique_ptr<IntermediateHitDoublets> intermediateHitDoublets;
  if(produceIntermediateHitDoublets_) {
    intermediateHitDoublets = std::make_unique<IntermediateHitDoublets>(&layers);
    intermediateHitDoublets->reserve(regions.size(), layers.size());
  }

  for(const TrackingRegion& region: regions) {
    if(produceIntermediateHitDoublets_) {
      intermediateHitDoublets->beginRegion(&region);
    }

    for(SeedingLayerSetsHits::SeedingLayerSet layerSet: layers) {
      LayerHitMapCache hitCache;
      auto doublets = generator_.doublets(region, iEvent, iSetup, layerSet, hitCache);
      if(doublets.empty()) continue; // don't bother if no pairs from these layers
      if(produceSeedingHitSets_) {
        for(size_t i=0, size=doublets.size(); i<size; ++i) {
          seedingHitSets->emplace_back(doublets.hit(i, HitDoublets::inner),
                                       doublets.hit(i, HitDoublets::outer));
        }
      }
      if(produceIntermediateHitDoublets_) {
        intermediateHitDoublets->addDoublets(layerSet, std::move(doublets), std::move(hitCache));
      }
    }
  }

  if(produceSeedingHitSets_) {
    seedingHitSets->shrink_to_fit();
    localRA_.update(seedingHitSets->size());
    iEvent.put(std::move(seedingHitSets));
  }
  if(produceIntermediateHitDoublets_) {
    intermediateHitDoublets->shrink_to_fit();
    iEvent.put(std::move(intermediateHitDoublets));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HitPairEDProducer);
