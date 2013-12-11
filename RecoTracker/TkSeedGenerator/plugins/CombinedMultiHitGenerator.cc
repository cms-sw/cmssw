#include "CombinedMultiHitGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace ctfseeding;

CombinedMultiHitGenerator::CombinedMultiHitGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
  : initialised(false), theConfig(cfg)
{ }

void CombinedMultiHitGenerator::init(const edm::ParameterSet & cfg, const edm::EventSetup& es)
{
//  edm::ParameterSet leyerPSet = cfg.getParameter<edm::ParameterSet>("LayerPSet");
//  SeedingLayerSets layerSets  = SeedingLayerSetsBuilder(leyerPSet).layers();

  std::string layerBuilderName = cfg.getParameter<std::string>("SeedingLayers");
  edm::ESHandle<SeedingLayerSetsBuilder> layerBuilder;
  es.get<TrackerDigiGeometryRecord>().get(layerBuilderName, layerBuilder);

  SeedingLayerSets layerSets  =  layerBuilder->layers();


  vector<LayerTriplets::LayerPairAndLayers>::const_iterator it;
  vector<LayerTriplets::LayerPairAndLayers> trilayers=LayerTriplets(layerSets).layers();

  for (it = trilayers.begin(); it != trilayers.end(); it++) {
    SeedingLayer first = (*it).first.first;
    SeedingLayer second = (*it).first.second;
    vector<SeedingLayer> thirds = (*it).second;

    edm::ParameterSet generatorPSet = theConfig.getParameter<edm::ParameterSet>("GeneratorPSet");
    std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");

    MultiHitGeneratorFromPairAndLayers * aGen =
        MultiHitGeneratorFromPairAndLayersFactory::get()->create(generatorName,generatorPSet);

    aGen->init( HitPairGeneratorFromLayerPair( first, second, &theLayerCache),
		thirds, &theLayerCache);

    theGenerators.push_back( aGen);
  }

  initialised = true;

}

CombinedMultiHitGenerator::~CombinedMultiHitGenerator()
{
  GeneratorContainer::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
}


void CombinedMultiHitGenerator::hitSets(
   const TrackingRegion& region, OrderedMultiHits & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  if (!initialised) init(theConfig,es);

  GeneratorContainer::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitSets( region, result, ev, es);
  }
  theLayerCache.clear();
}

