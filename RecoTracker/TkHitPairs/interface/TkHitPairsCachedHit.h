#ifndef TkHitPairsCachedHit_H
#define TkHitPairsCachedHit_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/Framework/interface/ESHandle.h"
class TkHitPairsCachedHit {
public:
  TkHitPairsCachedHit(const TrackingRecHit * hit ,  const edm::EventSetup& iSetup) : theRecHit(hit) {
    edm::ESHandle<TrackerGeometry> tracker;
    iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
    GlobalPoint gp = tracker->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition());
    thePhi = gp.phi(); 
    theR = gp.perp();
    theZ = gp.z();
    unsigned int subid=hit->geographicalId().subdetId();
    theRZ= (subid==  PixelSubdetector::PixelBarrel) ? theZ : theR;
  
  }
  float phi() const {return thePhi;}
  float rOrZ() const { return theRZ; } 
  float r() const {return theR; }
  float z() const {return theZ; }

  const TrackingRecHit * RecHit() const { return theRecHit;}
private:
  const TrackingRecHit *theRecHit;
  float thePhi;
  float theR, theZ;
  float theRZ;
};

#endif 
