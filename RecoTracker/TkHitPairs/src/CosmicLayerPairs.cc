#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"




vector<SeedLayerPairs::LayerPair> CosmicLayerPairs::operator()() 
{
  vector<LayerPair> result;

  //seeds from the barrel
  result.push_back( LayerPair(lh1,lh2));


  return result;
}

void CosmicLayerPairs::addBarrelBarrelLayers( int mid, int outer, 
						 vector<LayerPair>& result) const
{

 
}

void CosmicLayerPairs::init(const SiStripRecHit2DLocalPosCollection &collstereo,
			    const SiStripRecHit2DLocalPosCollection &collrphi, 
			    const edm::EventSetup& iSetup){

  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
  bl=track->barrelLayers(); 
  fpos=track->posForwardLayers(); 
  fneg=track->negForwardLayers(); 
  
  map_range1=collrphi.get(acc.stripTOBLayer(7));
  map_range2=collrphi.get(acc.stripTOBLayer(8));

  


   const TOBLayer*  bl1=dynamic_cast<TOBLayer*>(bl[11]);
   const TOBLayer*  bl2=dynamic_cast<TOBLayer*>(bl[12]);


//   //LayersWithHits

  lh1=new  LayerWithHits(bl1,map_range1);
  lh2=new  LayerWithHits(bl2,map_range2);


}
