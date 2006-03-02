#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"
//#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"

using namespace std;

void SeedGeneratorFromLayerPairs::initPairGenerator(
    const SeedLayerPairs * layerPairs) 
{
  if (layerPairs) { 
    thePairGenerator = new CombinedHitPairGenerator(*layerPairs);
  } else thePairGenerator = 0;
}

SeedGeneratorFromLayerPairs::SeedGeneratorFromLayerPairs(
    const SeedLayerPairs * layerPairs)
  : 
  //theRegionFactory(0), 
theRegion(0)
{ initPairGenerator(layerPairs); }


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

SeedGeneratorFromLayerPairs::~SeedGeneratorFromLayerPairs()
{ delete thePairGenerator; }

const TrackingRegion * SeedGeneratorFromLayerPairs::trackingRegion() const
{
  if (theRegion) return &(*theRegion);
  //  else if (theRegionFactory) return theRegionFactory->region();
  //  else throw DetLogicError("** SeedGeneratorFromLayerPairs **: cannot work without knowlege of region or region factory");
  else cerr<<"** SeedGeneratorFromLayerPairs **: cannot work without knowlege of region or region factory"<<endl;
}

HitPairGenerator * SeedGeneratorFromLayerPairs::pairGenerator() const
{
  if (!thePairGenerator) {
    //  throw DetLogicError("** SeedGeneratorFromLayerPairs **: pairGenerator() called but thePairGenerator is null!");
    cerr<< "** SeedGeneratorFromLayerPairs **: pairGenerator() called but thePairGenerator is null!"<<cerr;
  }
  else return thePairGenerator;
}

