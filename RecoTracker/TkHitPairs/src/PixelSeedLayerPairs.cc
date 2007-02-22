#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

using std::vector;

PixelSeedLayerPairs::~PixelSeedLayerPairs()
{
  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); it!=allLayersWithHits.end();it++){
    delete *it;
  }
}


vector<SeedLayerPairs::LayerPair> PixelSeedLayerPairs::operator()() 
{  
  vector<LayerPair> result;

  //seeds from the barrel
  result.push_back( LayerPair(lh1,lh2));
  result.push_back( LayerPair(lh1,lh3));
  result.push_back( LayerPair(lh2,lh3));

  //seeds from the forward-barrel
  result.push_back( LayerPair(lh1,pos1));
  result.push_back( LayerPair(lh1,neg1));
  result.push_back( LayerPair(lh1,pos2));
  result.push_back( LayerPair(lh1,neg2));
  result.push_back( LayerPair(lh2,pos1));
  result.push_back( LayerPair(lh2,neg1));
  result.push_back( LayerPair(lh2,pos2));
  result.push_back( LayerPair(lh2,neg2));

  //seeds from the forward
  result.push_back( LayerPair(pos1,pos2));
  result.push_back( LayerPair(neg1,neg2));

  return result;
}

void PixelSeedLayerPairs::init(const SiPixelRecHitCollection &coll,
			       const edm::EventSetup& iSetup)
{
  if(isFirstCall){
    edm::ESHandle<GeometricSearchTracker> track;
    iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
    bl=track->barrelLayers(); 
    fpos=track->posForwardLayers(); 
    fneg=track->negForwardLayers(); 
    isFirstCall=false;
  }
  
  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); it!=allLayersWithHits.end();it++){
    delete *it;
  }
  allLayersWithHits.clear();

  map_range1=coll.get(acc.pixelBarrelLayer(1));
  map_range2=coll.get(acc.pixelBarrelLayer(2));
  map_range3=coll.get(acc.pixelBarrelLayer(3));

  map_diskneg1=coll.get(acc.pixelForwardDisk(1,1));
  map_diskneg2=coll.get(acc.pixelForwardDisk(1,2));
  map_diskpos1=coll.get(acc.pixelForwardDisk(2,1));
  map_diskpos2=coll.get(acc.pixelForwardDisk(2,2));

  lh1=new  LayerWithHits(bl[0],map_range1); allLayersWithHits.push_back(lh1); 
  lh2=new  LayerWithHits(bl[1],map_range2); allLayersWithHits.push_back(lh2); 
  lh3=new  LayerWithHits(bl[2],map_range3); allLayersWithHits.push_back(lh3); 
  pos1=new  LayerWithHits(fpos[0],map_diskpos1); allLayersWithHits.push_back(pos1); 
  pos2=new  LayerWithHits(fpos[1],map_diskpos2); allLayersWithHits.push_back(pos2); 
  neg1=new  LayerWithHits(fneg[0],map_diskneg1); allLayersWithHits.push_back(neg1); 
  neg2=new  LayerWithHits(fneg[1],map_diskneg2); allLayersWithHits.push_back(neg2);
}

