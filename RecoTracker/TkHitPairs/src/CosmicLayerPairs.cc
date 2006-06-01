#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"

vector<SeedLayerPairs::LayerPair> CosmicLayerPairs::operator()() 
{
  vector<LayerPair> result;

  //seeds from the barrel
  result.push_back( LayerPair(lh1,lh2));


  return result;
}

void CosmicLayerPairs::init(const SiStripRecHit2DLocalPosCollection &collstereo,
			    const SiStripRecHit2DLocalPosCollection &collrphi, 
			    const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
			    const edm::EventSetup& iSetup){


  edm::ESHandle<GeometricSearchTracker> track;

  iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
  bl=track->barrelLayers(); 

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
//  SiStripRecHit2DLocalPosCollection::const_iterator istrip;
//   for(istrip=collrphi.begin();istrip!=collrphi.end();istrip++){
//     GlobalPoint gp = tracker->idToDet((*istrip).geographicalId())->surface().toGlobal((*istrip).localPosition());
//     DetId jj=(*istrip).geographicalId();
//     unsigned int subid=jj.subdetId();
//     int indexlayer=0;
//     if    (subid==  StripSubdetector::TIB)  indexlayer=TIBDetId(jj).layer();
    
//     if    (subid== StripSubdetector::TOB)  indexlayer=TOBDetId(jj).layer();
//     cout<<"DetId "<<(*istrip).geographicalId().rawId()<<" posizione "<<gp<<" subid "<< subid<<" layer "<<indexlayer<<endl;
//   }
  const TIBLayer*  bl1=dynamic_cast<TIBLayer*>(bl[0]);
  const TIBLayer*  bl2=dynamic_cast<TIBLayer*>(bl[1]);

  match_range1=collmatched.get(acc.stripTIBLayer(1));
  rphi_range2=collrphi.get(acc.stripTIBLayer(2));
  lh1=new  LayerWithHits(bl1,match_range1);
  lh2=new  LayerWithHits(bl2,rphi_range2);


}
