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
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"

class HitPairEDProducer: public edm::stream::EDProducer<> {
public:
  HitPairEDProducer(const edm::ParameterSet& iConfig);
  ~HitPairEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> seedingLayerToken_;
  edm::EDGetTokenT<edm::OwnVector<TrackingRegion> > regionToken_;
  edm::EDGetTokenT<bool> clusterCheckToken_;

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
  clusterCheckToken_(consumes<bool>(iConfig.getParameter<edm::InputTag>("clusterCheck"))),
  maxElement_(iConfig.getParameter<unsigned int>("maxElement")),
  generator_(0, 1, nullptr, maxElement_), // TODO: make layer indices configurable?
  produceSeedingHitSets_(iConfig.getParameter<bool>("produceSeedingHitSets")),
  produceIntermediateHitDoublets_(iConfig.getParameter<bool>("produceIntermediateHitDoublets"))
{
  if(!produceIntermediateHitDoublets_ && !produceSeedingHitSets_)
    throw cms::Exception("Configuration") << "HitPairEDProducer requires either produceIntermediateHitDoublets or produceSeedingHitSets to be True. If neither are needed, just remove this module from your sequence/path as it doesn't do anything useful";

  if(produceSeedingHitSets_)
    produces<RegionsSeedingHitSets>();
  if(produceIntermediateHitDoublets_)
    produces<IntermediateHitDoublets>();
}

void HitPairEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("seedingLayers", edm::InputTag("seedingLayersEDProducer"));
  desc.add<edm::InputTag>("trackingRegions", edm::InputTag("globalTrackingRegionFromBeamSpot"));
  desc.add<edm::InputTag>("clusterCheck", edm::InputTag("trackerClusterCheck"));
  desc.add<bool>("produceSeedingHitSets", false);
  desc.add<bool>("produceIntermediateHitDoublets", false);
  desc.add<unsigned int>("maxElement", 1000000);

  descriptions.add("hitPairEDProducer", desc);
}

void HitPairEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<bool> hclusterCheck;
  iEvent.getByToken(clusterCheckToken_, hclusterCheck);
  const bool clusterCheckOk = *hclusterCheck;

  edm::Handle<SeedingLayerSetsHits> hlayers;
  iEvent.getByToken(seedingLayerToken_, hlayers);
  const auto& layers = *hlayers;
  if(layers.numberOfLayersInSet() < 2)
    throw cms::Exception("Configuration") << "HitPairEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 2, got " << layers.numberOfLayersInSet();

  edm::Handle<edm::OwnVector<TrackingRegion> > hregions;
  iEvent.getByToken(regionToken_, hregions);
  const auto& regions = *hregions;

  std::unique_ptr<RegionsSeedingHitSets> seedingHitSets;
  std::unique_ptr<IntermediateHitDoublets> intermediateHitDoublets;
  if(produceSeedingHitSets_)
    seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if(produceIntermediateHitDoublets_)
    intermediateHitDoublets = std::make_unique<IntermediateHitDoublets>(&layers);

  if(!clusterCheckOk) {
    if(produceSeedingHitSets_)
      iEvent.put(std::move(seedingHitSets));
    if(produceIntermediateHitDoublets_)
      iEvent.put(std::move(intermediateHitDoublets));
    return;
  }

  if(produceSeedingHitSets_)
    seedingHitSets->reserve(regions.size(), localRA_.upper());
  if(produceIntermediateHitDoublets_)
    intermediateHitDoublets->reserve(regions.size(), layers.size());


  LogDebug("HitPairEDProducer") << "Creating doublets for " << regions.size() << " and " << layers.size() << " layer sets";

  // This is the easiest way to extract the layer pairs from the full
  // set of seeding layers. It feels a bit stupid to do it for each
  // event given the input is defined in configuration. Maybe do it
  // once-per-job in SeedingLayerSetsEDProducer?
  std::vector<SeedingLayerSetsHits::SeedingLayerSet> layerPairs;
  if(layers.numberOfLayersInSet() > 2) {
    std::vector<LayerTriplets::LayerSetAndLayers> trilayers = LayerTriplets::layers(layers);
    layerPairs.reserve(trilayers.size());
    for(const auto& setAndLayers: trilayers) {
      layerPairs.push_back(setAndLayers.first);
    }
  }
  else {
    layerPairs.reserve(layers.size());
    for(const auto& set: layers)
      layerPairs.push_back(set);
  }

  LayerHitMapCache hitCacheTmp; // used if !produceIntermediateHitDoublets_
  for(const TrackingRegion& region: regions) {
    auto seedingHitSetsFiller = RegionsSeedingHitSets::dummyFiller();
    auto intermediateHitDoubletsFiller = IntermediateHitDoublets::dummyFiller();
    auto hitCachePtr = &hitCacheTmp;
    if(produceSeedingHitSets_) {
      seedingHitSetsFiller = seedingHitSets->beginRegion(&region);
    }
    if(produceIntermediateHitDoublets_) {
      intermediateHitDoubletsFiller = intermediateHitDoublets->beginRegion(&region);
      hitCachePtr = &(intermediateHitDoubletsFiller.layerHitMapCache());
    }
    else {
      hitCacheTmp.clear();
    }


    for(SeedingLayerSetsHits::SeedingLayerSet layerSet: layerPairs) {
      auto doublets = generator_.doublets(region, iEvent, iSetup, layerSet, *hitCachePtr);
      LogTrace("HitPairEDProducer") << " created " << doublets.size() << " doublets for layers " << layerSet[0].index() << "," << layerSet[1].index();
      if(doublets.empty()) continue; // don't bother if no pairs from these layers
      if(produceSeedingHitSets_) {
        for(size_t i=0, size=doublets.size(); i<size; ++i) {
          seedingHitSetsFiller.emplace_back(doublets.hit(i, HitDoublets::inner),
                                            doublets.hit(i, HitDoublets::outer));
        }
      }
      if(produceIntermediateHitDoublets_) {
        intermediateHitDoubletsFiller.addDoublets(layerSet, std::move(doublets));
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
