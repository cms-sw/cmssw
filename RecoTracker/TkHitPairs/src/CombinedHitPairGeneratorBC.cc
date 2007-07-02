//#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGeneratorBC.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPairBC.h"

using namespace std;


//CombinedHitPairGeneratorBC::CombinedHitPairGeneratorBC(SeedLayerPairs& layers)
CombinedHitPairGeneratorBC::CombinedHitPairGeneratorBC(SeedLayerPairs& layers,
						   const edm::EventSetup& iSetup)
{
  vector<SeedLayerPairs::LayerPair> layerPairs = layers();
  vector<SeedLayerPairs::LayerPair>::const_iterator it;
  for (it = layerPairs.begin(); it != layerPairs.end(); it++) {
    add( (*it).first, (*it).second,iSetup);
  }

}

// CombinedHitPairGeneratorBC::CombinedHitPairGeneratorBC(
//     const CombinedHitPairGeneratorBC &o)
// {
//   typedef Container::const_iterator IC;
//   for (IC i = o.theGenerators.begin(); i != o.theGenerators.end(); i++) 
//       add( (**i).innerLayer(), (**i).outerLayer() );
// }

CombinedHitPairGeneratorBC::~CombinedHitPairGeneratorBC()
{
  Container::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
}

// void CombinedHitPairGeneratorBC::add(
//     const DetLayer* inner, const DetLayer* outer) 
void CombinedHitPairGeneratorBC::add(
				   const LayerWithHits *inner, const LayerWithHits *outer,
				   const edm::EventSetup& iSetup) 
{ 

  theGenerators.push_back( 
			  new HitPairGeneratorFromLayerPairBC( inner, outer, &theLayerCache,iSetup));
}

void CombinedHitPairGeneratorBC::hitPairs(
					const TrackingRegion& region, 
					OrderedHitPairs & pairs,
					const edm::EventSetup& iSetup)
{
 
  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
 
    (**i).hitPairs( region, pairs, iSetup); 
  }
 
  theLayerCache.clear();
 
}


