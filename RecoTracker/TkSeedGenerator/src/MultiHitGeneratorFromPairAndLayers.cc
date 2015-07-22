#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

MultiHitGeneratorFromPairAndLayers::MultiHitGeneratorFromPairAndLayers(const edm::ParameterSet& pset):
  theLayerCache(nullptr),
  theMaxElement(pset.getParameter<unsigned int>("maxElement"))
{}

MultiHitGeneratorFromPairAndLayers::~MultiHitGeneratorFromPairAndLayers() {}

void MultiHitGeneratorFromPairAndLayers::init(std::unique_ptr<HitPairGeneratorFromLayerPair>&& pairGenerator, LayerCacheType *layerCache) {
  thePairGenerator = std::move(pairGenerator);
  theLayerCache = layerCache;
}

void MultiHitGeneratorFromPairAndLayers::clear() {
  cache.clear();
}
