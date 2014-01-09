#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

CombinedHitPairGenerator::CombinedHitPairGenerator(const edm::ParameterSet& cfg)
  : theSeedingLayerSrc(cfg.getParameter<edm::InputTag>("SeedingLayers"))
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
  theGenerator.reset(new HitPairGeneratorFromLayerPair(0, 1, &theLayerCache, 0, theMaxElement));
}

CombinedHitPairGenerator::CombinedHitPairGenerator(const CombinedHitPairGenerator& cb):
  theSeedingLayerSrc(cb.theSeedingLayerSrc),
  theGenerator(new HitPairGeneratorFromLayerPair(0, 1, &theLayerCache, 0, cb.theMaxElement))
{
  theMaxElement = cb.theMaxElement;
}

CombinedHitPairGenerator::~CombinedHitPairGenerator() {}

void CombinedHitPairGenerator::setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet layers) {
  assert(0 == "not implemented");
}

void CombinedHitPairGenerator::hitPairs(
   const TrackingRegion& region, OrderedHitPairs  & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByLabel(theSeedingLayerSrc, hlayers);
  const SeedingLayerSetsHits& layers = *hlayers;
  if(layers.numberOfLayersInSet() != 2)
    throw cms::Exception("Configuration") << "CombinedHitPairGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 2, got " << layers.numberOfLayersInSet();

  for(SeedingLayerSetsHits::SeedingLayerSet layerSet: layers) {
    theGenerator->setSeedingLayers(layerSet);
    theGenerator->hitPairs( region, result, ev, es);
  }

  theLayerCache.clear();

  LogDebug("CombinedHitPairGenerator")<<" total number of pairs provided back CHPG : "<<result.size();

}
