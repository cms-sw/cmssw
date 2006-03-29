#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// PixelSeedLayerPairs::PixelSeedLayerPairs()
// {
//  //  TrackerLayerIdAccessor accessor;
// //   theBarrelPixel = accessor.pixelBarrelLayers();
// //   theNegPixel = accessor.pixelNegativeForwardLayers();
// //   thePosPixel = accessor.pixelPositiveForwardLayers();
// }

//#define DEBUG


#ifdef DEBUG
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#endif

vector<SeedLayerPairs::LayerPair> PixelSeedLayerPairs::operator()() 
{
  vector<LayerPair> result;

  //seeds from the barrel
  result.push_back( LayerPair(lh1,lh2));
  result.push_back( LayerPair(lh1,lh3));
  result.push_back( LayerPair(lh2,lh3));
  //seeds from the forward
  result.push_back( LayerPair(pos1,pos2));
  result.push_back( LayerPair(neg1,neg2));
  //seeds from the forward-barrel
  result.push_back( LayerPair(lh1,pos1));
  result.push_back( LayerPair(lh1,neg1));
  result.push_back( LayerPair(lh1,pos2));
  result.push_back( LayerPair(lh1,neg2));
  result.push_back( LayerPair(lh2,pos2));
  result.push_back( LayerPair(lh2,neg2));

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

  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
  bl=track->barrelLayers(); 
  fpos=track->posForwardLayers(); 
  fneg=track->negForwardLayers(); 
  
  map_range1=coll.get(acc.pixelBarrelLayer(1));
  map_range2=coll.get(acc.pixelBarrelLayer(2));
  map_range3=coll.get(acc.pixelBarrelLayer(3));

  //MP check the side
  map_diskneg1=coll.get(acc.pixelForwardDisk(1,1));
  map_diskneg2=coll.get(acc.pixelForwardDisk(1,2));
  map_diskpos1=coll.get(acc.pixelForwardDisk(2,1));
  map_diskpos2=coll.get(acc.pixelForwardDisk(2,2));
  
#ifdef DEBUG

  SiPixelRecHitCollection::range map_diskpos3;
  SiPixelRecHitCollection::range map_diskneg3;
  map_diskneg3=coll.get(acc.pixelForwardDisk(1,3));
  map_diskpos3=coll.get(acc.pixelForwardDisk(2,3));


  for (SiPixelRecHitCollection::const_iterator it = coll.begin(); it != coll.end(); it++){
    unsigned int id((*it).geographicalId().rawId());
    DetId ii(id);
    if (ii.subdetId() == PixelSubdetector::PixelBarrel){
      PXBDetId iii(id);
      std::cout <<" Pixel Hit on barrel "<<iii.layer()<<std::endl;
    }else{
      PXFDetId iii(id);
      LogDebug("PixelSeedLayerPairs") <<" Pixel Hit on Disk "<<iii.disk()<<" " <<iii.side();
    }
  }
   unsigned int tot = coll.end()-coll.begin();
   unsigned int b1 = map_range1.second-map_range1.first;
   unsigned int b2 = map_range2.second-map_range2.first;
   unsigned int b3 = map_range3.second-map_range3.first;
   unsigned int fp1 = map_diskpos1.second-map_diskpos1.first;
   unsigned int fp2 = map_diskpos2.second-map_diskpos2.first;
   unsigned int fp3 = map_diskpos3.second-map_diskpos3.first;
   unsigned int fn1 = map_diskneg1.second-map_diskneg1.first;
   unsigned int fn2 = map_diskneg2.second-map_diskneg2.first;
   unsigned int fn3 = map_diskneg3.second-map_diskneg3.first;
   LogDebug("PixelSeedLayerPairs")<<"Total Number of Pixel RecHits "<<tot;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits B1 "<<b1;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits B2 "<<b2;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits B3 "<<b3;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits FP1 "<<fp1;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits FP2 "<<fp2;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits FP3 "<<fp3;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits FN1 "<<fn1;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits FN2 "<<fn2;
   LogDebug("PixelSeedLayerPairs")<<"Number of Pixel RecHits FN3 "<<fn3;
   int res = tot-b1-b2-b3-fp1-fp2-fp3-fn1-fn2-fn3;
   LogDebug("PixelSeedLayerPairs") <<" Total number of PixelRecHits "<<tot;
  
#endif

  //BarrelLayers
  const PixelBarrelLayer*  bl1=dynamic_cast<PixelBarrelLayer*>(bl[0]);
  const PixelBarrelLayer*  bl2=dynamic_cast<PixelBarrelLayer*>(bl[1]);
  const PixelBarrelLayer*  bl3=dynamic_cast<PixelBarrelLayer*>(bl[2]);
  //ForwardLayers
  const PixelForwardLayer*  fpos1=dynamic_cast<PixelForwardLayer*>(fpos[0]);
  const PixelForwardLayer*  fpos2=dynamic_cast<PixelForwardLayer*>(fpos[1]);
  const PixelForwardLayer*  fneg1=dynamic_cast<PixelForwardLayer*>(fneg[0]);
  const PixelForwardLayer*  fneg2=dynamic_cast<PixelForwardLayer*>(fneg[1]);

  //LayersWithHits

  lh1=new  LayerWithHits(bl1,map_range1);
  lh2=new  LayerWithHits(bl2,map_range2);
  lh3=new  LayerWithHits(bl3,map_range3);
  
  pos1=new  LayerWithHits(fpos1,map_diskpos1);
  pos2=new  LayerWithHits(fpos2,map_diskpos2);
  neg1=new  LayerWithHits(fneg1,map_diskneg1);
  neg2=new  LayerWithHits(fneg2,map_diskneg2);

}
