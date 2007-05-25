#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

using namespace std;
using namespace ctfseeding;

CombinedHitPairGenerator::CombinedHitPairGenerator(const edm::ParameterSet& cfg)
  : initialised(false), theConfig(cfg)
{ }

CombinedHitPairGenerator::CombinedHitPairGenerator(const SeedingLayerSets & layerSets)
  : initialised(false)
{
  init (layerSets);
}

void CombinedHitPairGenerator::init(const edm::ParameterSet & cfg, const edm::EventSetup& es)
{
  edm::ParameterSet leyerPSet = cfg.getParameter<edm::ParameterSet>("LayerPSet");
  SeedingLayerSets layerSets  = SeedingLayerSetsBuilder(leyerPSet).layers(es);
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

CombinedHitPairGenerator::~CombinedHitPairGenerator()
{
  Container::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
}

void CombinedHitPairGenerator::add( const SeedingLayer& inner, const SeedingLayer& outer)
{ 
  theGenerators.push_back( new HitPairGeneratorFromLayerPair( inner, outer, &theLayerCache));
}

void CombinedHitPairGenerator::hitPairs(
   const TrackingRegion& region, OrderedHitPairs  & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  if (!initialised) init(theConfig,es);

  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitPairs( region, result, ev, es); 
  }
  theLayerCache.clear();
}
