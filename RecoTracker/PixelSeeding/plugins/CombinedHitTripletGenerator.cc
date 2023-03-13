#include "CombinedHitTripletGenerator.h"

#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoTracker/PixelSeeding/interface/LayerTriplets.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

CombinedHitTripletGenerator::CombinedHitTripletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers"))) {
  edm::ParameterSet generatorPSet = cfg.getParameter<edm::ParameterSet>("GeneratorPSet");
  std::string generatorName = generatorPSet.getParameter<std::string>("ComponentName");
  theGenerator = HitTripletGeneratorFromPairAndLayersFactory::get()->create(generatorName, generatorPSet, iC);
  theGenerator->init(std::make_unique<HitPairGeneratorFromLayerPair>(iC, 0, 1, &theLayerCache), &theLayerCache);
}

CombinedHitTripletGenerator::~CombinedHitTripletGenerator() {}

void CombinedHitTripletGenerator::hitTriplets(const TrackingRegion& region,
                                              OrderedHitTriplets& result,
                                              const edm::Event& ev,
                                              const edm::EventSetup& es) {
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByToken(theSeedingLayerToken, hlayers);
  const SeedingLayerSetsHits& layers = *hlayers;
  if (layers.numberOfLayersInSet() != 3)
    throw cms::Exception("Configuration")
        << "CombinedHitTripletGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 3, got "
        << layers.numberOfLayersInSet();

  std::vector<LayerTriplets::LayerSetAndLayers> trilayers = LayerTriplets::layers(layers);
  for (const auto& setAndLayers : trilayers) {
    theGenerator->hitTriplets(region, result, ev, es, setAndLayers.first, setAndLayers.second);
  }
  theLayerCache.clear();
}
