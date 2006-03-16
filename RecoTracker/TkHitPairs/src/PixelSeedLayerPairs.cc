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
 

  


  LayerPair tt=LayerPair(lh1,lh2);
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

void PixelSeedLayerPairs::init(const SiPixelRecHitCollection& coll, const edm::EventSetup& iSetup){
  edm::ESHandle<TrackingGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);


 
  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
  bl=track->barrelLayers();
 
  //map ranges
//   map_range1=coll.get(acc.pixelBarrelLayer(1));
//   map_range2=coll.get(acc.pixelBarrelLayer(2));
//   map_range3=coll.get(acc.pixelBarrelLayer(3));
  lay1=acc.pixelBarrelLayer(1);
  lay2=acc.pixelBarrelLayer(2);
  lay3=acc.pixelBarrelLayer(3);
  map_range1=coll.get(lay1.first,DetIdPXBSameLayerComparator());
  map_range2=coll.get(lay2.first,DetIdPXBSameLayerComparator());
  map_range3=coll.get(lay3.first,DetIdPXBSameLayerComparator());

 
  //BarrelLayers
  const PixelBarrelLayer*  bl1=dynamic_cast<PixelBarrelLayer*>(bl[0]);
  const PixelBarrelLayer*  bl2=dynamic_cast<PixelBarrelLayer*>(bl[1]);

  //LayersWithHits
 
  lh1=new  LayerWithHits(bl1,map_range1);
  lh2=new  LayerWithHits(bl2,map_range2);
 
  //  map_range2=coll.get(lay2.first,lay2.second);
  //  map_range3=coll.get(lay3.first,lay3.second);


}
