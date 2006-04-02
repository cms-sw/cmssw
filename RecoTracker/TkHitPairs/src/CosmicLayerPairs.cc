#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"



vector<SeedLayerPairs::LayerPair> CosmicLayerPairs::operator()() 
{
  vector<LayerPair> result;

  //seeds from the barrel
  result.push_back( LayerPair(lh1,lh2));


  return result;
}

void CosmicLayerPairs::init(const SiStripRecHit2DLocalPosCollection &collstereo,
			    const SiStripRecHit2DLocalPosCollection &collrphi, 
			    const edm::EventSetup& iSetup){

  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
  bl=track->barrelLayers(); 
  fpos=track->posForwardLayers(); 
  fneg=track->negForwardLayers(); 
  
  rphi_range1=collrphi.get(acc.stripTOBLayer(5));
  rphi_range2=collrphi.get(acc.stripTOBLayer(6));


  const TOBLayer*  bl1=dynamic_cast<TOBLayer*>(bl[11]);
  const TOBLayer*  bl2=dynamic_cast<TOBLayer*>(bl[12]);

  

//   //LayersWithHits

  lh1=new  LayerWithHits(bl1,rphi_range1);
  lh2=new  LayerWithHits(bl2,rphi_range2);


}
