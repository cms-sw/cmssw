#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

HitTripletGeneratorFromPairAndLayers::HitTripletGeneratorFromPairAndLayers(unsigned int maxElement)
    : theLayerCache(nullptr), theMaxElement(maxElement) {}

HitTripletGeneratorFromPairAndLayers::HitTripletGeneratorFromPairAndLayers(const edm::ParameterSet& pset)
    : HitTripletGeneratorFromPairAndLayers(pset.getParameter<unsigned int>("maxElement")) {}

HitTripletGeneratorFromPairAndLayers::~HitTripletGeneratorFromPairAndLayers() {}

void HitTripletGeneratorFromPairAndLayers::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<unsigned int>("maxElement", 1000000);
}

void HitTripletGeneratorFromPairAndLayers::init(std::unique_ptr<HitPairGeneratorFromLayerPair>&& pairGenerator,
                                                LayerCacheType* layerCache) {
  thePairGenerator = std::move(pairGenerator);
  theLayerCache = layerCache;
}
