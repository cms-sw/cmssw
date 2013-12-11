#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace ctfseeding;

CombinedHitPairGenerator::CombinedHitPairGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
{
  unsigned maxElement = cfg.getParameter<unsigned int>("maxElement");
  SeedingLayerSetsBuilder layerBuilder(cfg.getParameter<edm::ParameterSet>("SeedingLayers"), iC);

  SeedingLayerSets layerSets  =  layerBuilder.layers();

  typedef SeedingLayerSets::const_iterator IL;
  for (IL il=layerSets.begin(), ilEnd=layerSets.end(); il != ilEnd; ++il) {
    const SeedingLayers & set = *il;
    if (set.size() != 2) continue;
    theGenerators.emplace_back( new HitPairGeneratorFromLayerPair( set[0], set[1], &theLayerCache, 0, maxElement));
  }
}

CombinedHitPairGenerator::CombinedHitPairGenerator(const CombinedHitPairGenerator& gen)
{
  theGenerators.reserve(gen.theGenerators.size());
  for(const auto& g: gen.theGenerators) {
    theGenerators.emplace_back(g->clone());
  }
}

CombinedHitPairGenerator::~CombinedHitPairGenerator() {}

CombinedHitPairGenerator *CombinedHitPairGenerator::clone() const {
  return new CombinedHitPairGenerator(*this);
}

void CombinedHitPairGenerator::hitPairs(
   const TrackingRegion& region, OrderedHitPairs  & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitPairs( region, result, ev, es); 
  }
  theLayerCache.clear();

  LogDebug("CombinedHitPairGenerator")<<" total number of pairs provided back CHPG : "<<result.size();

}
