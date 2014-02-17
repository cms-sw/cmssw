//#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGeneratorFromLayerPair.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;


CosmicHitPairGenerator::CosmicHitPairGenerator(SeedLayerPairs& layers,
						   const edm::EventSetup& iSetup)
{

  vector<SeedLayerPairs::LayerPair> layerPairs = layers();
  vector<SeedLayerPairs::LayerPair>::const_iterator it;
  for (it = layerPairs.begin(); it != layerPairs.end(); it++) {
    add( (*it).first, (*it).second,iSetup);
  }

}



CosmicHitPairGenerator::~CosmicHitPairGenerator()
{
}


void CosmicHitPairGenerator::add(
				   const LayerWithHits *inner, const LayerWithHits *outer,
				   const edm::EventSetup& iSetup) 
{ 
  theGenerators.push_back(std::make_unique<CosmicHitPairGeneratorFromLayerPair>( inner, outer, iSetup));
}

void CosmicHitPairGenerator::hitPairs(
					const TrackingRegion& region, 
					OrderedHitPairs & pairs,
					const edm::EventSetup& iSetup)
{

  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitPairs( region, pairs, iSetup); 
  }
 
}


