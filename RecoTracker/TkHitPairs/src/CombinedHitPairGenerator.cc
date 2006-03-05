//#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"




//CombinedHitPairGenerator::CombinedHitPairGenerator(SeedLayerPairs& layers)
CombinedHitPairGenerator::CombinedHitPairGenerator(SeedLayerPairs& layers,
						   const edm::EventSetup& iSetup)
{
  
  vector<SeedLayerPairs::LayerPair> layerPairs = layers();
  vector<SeedLayerPairs::LayerPair>::const_iterator it;
  for (it = layerPairs.begin(); it != layerPairs.end(); it++) {
    add( (*it).first, (*it).second);
  }
}

// CombinedHitPairGenerator::CombinedHitPairGenerator(
//     const CombinedHitPairGenerator &o)
// {
//   typedef Container::const_iterator IC;
//   for (IC i = o.theGenerators.begin(); i != o.theGenerators.end(); i++) 
//       add( (**i).innerLayer(), (**i).outerLayer() );
// }

CombinedHitPairGenerator::~CombinedHitPairGenerator()
{
  Container::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
}

// void CombinedHitPairGenerator::add(
//     const DetLayer* inner, const DetLayer* outer) 
void CombinedHitPairGenerator::add(
     const LayerWithHits *inner, const LayerWithHits *outer) 
{ 
  theGenerators.push_back( 
      new HitPairGeneratorFromLayerPair( inner, outer, &theLayerCache));
}

void CombinedHitPairGenerator::hitPairs(
    const TrackingRegion& region, OrderedHitPairs & pairs)
{
  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) 
    (**i).hitPairs( region, pairs); 
  theLayerCache.clear();
}


