#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"
//#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGeneratorBC.h"

using namespace std;

void SeedGeneratorFromLayerPairs::initPairGenerator(SeedLayerPairs * layerPairs,
						    const edm::EventSetup& iSetup) 
{
  if (layerPairs) { 
    thePairGenerator.reset(new CombinedHitPairGeneratorBC(*layerPairs,iSetup));
  } else thePairGenerator.reset(0);
}

//SeedGeneratorFromLayerPairs::SeedGeneratorFromLayerPairs(
//    const SeedLayerPairs * layerPairs)
SeedGeneratorFromLayerPairs::SeedGeneratorFromLayerPairs(SeedLayerPairs * layerPairs,
							 const edm::ParameterSet& conf)
  :  SeedGeneratorFromHitPairsConsecutiveHits(conf),
     //theRegionFactory(0), 
     thePairGenerator(0),
     theRegion(0){}


// SeedGeneratorFromLayerPairs::SeedGeneratorFromLayerPairs(
//   const SeedLayerPairs * layerPairs,
//   const TrackingRegionFactory& regionFactory)
//   : theRegionFactory(&regionFactory), theRegion(0) 
// { initPairGenerator(layerPairs); }

// SeedGeneratorFromLayerPairs::SeedGeneratorFromLayerPairs(
//   const SeedLayerPairs * layerPairs,
//   const TrackingRegion& region)
//   : theRegionFactory(0), theRegion( region.clone())
// {  initPairGenerator(layerPairs); }

// SeedGeneratorFromLayerPairs::SeedGeneratorFromLayerPairs(
//   const SeedLayerPairs * layerPairs,
//   float ptMin, float originRadius, float originHalfLength, float originZPos) 
//   : theRegionFactory(0)
// {
//   initPairGenerator(layerPairs);
//   theRegion = new GlobalTrackingRegion( ptMin, originRadius,
//                                         originHalfLength, originZPos);
// }


const TrackingRegion * SeedGeneratorFromLayerPairs::trackingRegion() const
{
  if (theRegion) return &(*theRegion);
  //  else if (theRegionFactory) return theRegionFactory->region();
  //  else throw DetLogicError("** SeedGeneratorFromLayerPairs **: cannot work without knowlege of region or region factory");
  else{ cerr<<"** SeedGeneratorFromLayerPairs **: cannot work without knowlege of region or region factory"<<endl;
    return 0;
  }
}

HitPairGenerator * SeedGeneratorFromLayerPairs::pairGenerator() const
{
  if (!thePairGenerator.get()) {
    //  throw DetLogicError("** SeedGeneratorFromLayerPairs **: pairGenerator() called but thePairGenerator is null!");
    cerr<< "** SeedGeneratorFromLayerPairs **: pairGenerator() called but thePairGenerator is null!"<<cerr;
    return 0;
  }
  else return thePairGenerator.get();
}

