#ifndef TkHitPairsCachedHit_H
#define TkHitPairsCachedHit_H

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
class TkHitPairsCachedHit {
public:
  TkHitPairsCachedHit(const SiPixelRecHit * hit ,  const edm::EventSetup& iSetup) : theRecHit(hit) {
    edm::ESHandle<TrackingGeometry> tracker;
    iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
    GlobalPoint gp = tracker->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition());
    thePhi = gp.phi(); 
    theR = gp.perp();
    theZ = gp.z();
    //MP
    unsigned int subid=hit->geographicalId().subdetId();
    theRZ= (subid==  PixelSubdetector::PixelBarrel) ? theZ : theR;
   // theRZ  = (hit.layer()->part()==barrel) ? theZ : theR;
   theRZ  = theZ ;

  }
  float phi() const {return thePhi;}
  float rOrZ() const { return theRZ; } 
  float r() const {return theR; }
  float z() const {return theZ; }

  const SiPixelRecHit*  RecHit() const { return theRecHit;}
  //  operator RecHit() const { return theRecHit;}
private:
  const SiPixelRecHit *theRecHit;
  float thePhi;
  float theR, theZ;
  float theRZ;
};

#endif 
