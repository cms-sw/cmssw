#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
// PixelSeedLayerPairs::PixelSeedLayerPairs()
// {
//  //  TrackerLayerIdAccessor accessor;
// //   theBarrelPixel = accessor.pixelBarrelLayers();
// //   theNegPixel = accessor.pixelNegativeForwardLayers();
// //   thePosPixel = accessor.pixelPositiveForwardLayers();
// }




//vector<SeedLayerPairs::LayerPair> PixelSeedLayerPairs::operator()() const
vector<SeedLayerPairs::LayerPair> PixelSeedLayerPairs::operator()() 
{
  vector<LayerPair> result;

  lay1 = acc.pixelBarrelLayer(1);  
  map_range1=map.get(lay1.first,lay1.second); 
  lay2 = acc.pixelBarrelLayer(2);  
  map_range2=map.get(lay2.first,lay2.second);
  PixelBarrelLayer* bl1 =dynamic_cast<PixelBarrelLayer*>(bl[0]);
  const LayerWithHits lh1=LayerWithHits(bl1,map_range1);
  PixelBarrelLayer* bl2 =dynamic_cast<PixelBarrelLayer*>(bl[1]);
  const  LayerWithHits lh2=LayerWithHits(bl2,map_range2);
  LayerPair tt=LayerPair(&lh1,&lh2);
  result.push_back( tt);
  // addBarrelBarrelLayers(0,1,result);

  return result;
}

void PixelSeedLayerPairs::addBarrelBarrelLayers( int mid, int outer, 
						 vector<LayerPair>& result) const
{





  //MP
  //  result.push_back( LayerPair(LayerWithHits(bl[mid],lay1) , LayerWithHits(bl[outer],lay2)));
}

// void PixelSeedLayerPairs::addBarrelForwardLayers( int mid, int outer, 
// 						  vector<LayerPair>& result) const
// {
//   result.push_back( LayerPair( theBarrelPixel[mid], theNegPixel[outer]));
//   result.push_back( LayerPair( theBarrelPixel[mid], thePosPixel[outer]));
// }

// void PixelSeedLayerPairs::addForwardForwardLayers( int mid, int outer, 
// 						   vector<LayerPair>& result) const
// {
//   result.push_back( LayerPair( theNegPixel[mid], theNegPixel[outer]));
//   result.push_back( LayerPair( thePosPixel[mid], thePosPixel[outer]));
// }

void PixelSeedLayerPairs::init(const edm::EventSetup& iSetup){
  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
  bl=track->barrelLayers();
  }
