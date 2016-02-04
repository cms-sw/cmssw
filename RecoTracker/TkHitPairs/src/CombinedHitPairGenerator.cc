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

CombinedHitPairGenerator::CombinedHitPairGenerator(const edm::ParameterSet& cfg)
  : initialised(false), theConfig(cfg)
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
}

void CombinedHitPairGenerator::init(const edm::ParameterSet & cfg, const edm::EventSetup& es)
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");

  std::string layerBuilderName = cfg.getParameter<std::string>("SeedingLayers");
  edm::ESHandle<SeedingLayerSetsBuilder> layerBuilder;
  es.get<TrackerDigiGeometryRecord>().get(layerBuilderName, layerBuilder);

  SeedingLayerSets layerSets  =  layerBuilder->layers(es); 
  init(layerSets);
}

void CombinedHitPairGenerator::init(const SeedingLayerSets & layerSets)
{
  initialised = true;
  typedef SeedingLayerSets::const_iterator IL;
  for (IL il=layerSets.begin(), ilEnd=layerSets.end(); il != ilEnd; ++il) {
    const SeedingLayers & set = *il;
    if (set.size() != 2) continue;
    add( set[0], set[1] );
  }
}

void CombinedHitPairGenerator::cleanup()
{
  Container::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
  theGenerators.clear();
}

CombinedHitPairGenerator::~CombinedHitPairGenerator() { cleanup(); }

void CombinedHitPairGenerator::add( const SeedingLayer& inner, const SeedingLayer& outer)
{ 
  theGenerators.push_back( new HitPairGeneratorFromLayerPair( inner, outer, &theLayerCache, 0, theMaxElement));
}

void CombinedHitPairGenerator::hitPairs(
   const TrackingRegion& region, OrderedHitPairs  & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  if (theESWatcher.check(es) || !initialised ) {
    cleanup();
    init(theConfig,es);
  }

  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitPairs( region, result, ev, es); 
  }
  theLayerCache.clear();

  LogDebug("CombinedHitPairGenerator")<<" total number of pairs provided back CHPG : "<<result.size();

}
