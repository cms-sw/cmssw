//#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGeneratorFromLayerPair.h"




//CosmicHitPairGenerator::CosmicHitPairGenerator(SeedLayerPairs& layers)
CosmicHitPairGenerator::CosmicHitPairGenerator(SeedLayerPairs& layers,
						   const edm::EventSetup& iSetup)
{
  vector<SeedLayerPairs::LayerPair> layerPairs = layers();
  vector<SeedLayerPairs::LayerPair>::const_iterator it;
  for (it = layerPairs.begin(); it != layerPairs.end(); it++) {
    add( (*it).first, (*it).second,iSetup);
  }

}

// CosmicHitPairGenerator::CosmicHitPairGenerator(
//     const CosmicHitPairGenerator &o)
// {
//   typedef Container::const_iterator IC;
//   for (IC i = o.theGenerators.begin(); i != o.theGenerators.end(); i++) 
//       add( (**i).innerLayer(), (**i).outerLayer() );
// }

CosmicHitPairGenerator::~CosmicHitPairGenerator()
{
  Container::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
}

// void CosmicHitPairGenerator::add(
//     const DetLayer* inner, const DetLayer* outer) 
void CosmicHitPairGenerator::add(
				   const LayerWithHits *inner, const LayerWithHits *outer,
				   const edm::EventSetup& iSetup) 
{ 

  theGenerators.push_back( 
			  new CosmicHitPairGeneratorFromLayerPair( inner, outer, &theLayerCache,iSetup));
}

void CosmicHitPairGenerator::hitPairs(
					const TrackingRegion& region, 
					OrderedHitPairs & pairs,
					const edm::EventSetup& iSetup)
{
 
  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
 
    (**i).hitPairs( region, pairs,iSetup); 
  }
 
  theLayerCache.clear();
 
}


