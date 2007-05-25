#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

using namespace ctfseeding;

class SeedingHit::SeedingHitImpl {
public:
  
  // obsolete - remove asap!
  SeedingHitImpl(const TrackingRecHit * hit ,  const edm::EventSetup& es) 
    : theRecHit(hit), hasTransientHit(false)
  {
    edm::ESHandle<TrackerGeometry> tracker;
    es.get<TrackerDigiGeometryRecord>().get(tracker);
    GlobalPoint gp = tracker->idToDet(
        hit->geographicalId())->surface().toGlobal(hit->localPosition());
    thePhi = gp.phi();
    theR = gp.perp();
    theZ = gp.z();
    unsigned int subid=hit->geographicalId().subdetId();
    theRZ = (   subid == PixelSubdetector::PixelBarrel
             || subid == StripSubdetector::TIB
             || subid == StripSubdetector::TOB) ? theZ : theR;
  }


  SeedingHitImpl(const TrackingRecHit * hit, const SeedingLayer &layer, const edm::EventSetup& es)
    : theLayer(layer), theRecHit(hit), hasTransientHit(false)  
  {
    edm::ESHandle<TrackerGeometry> tracker;
    es.get<TrackerDigiGeometryRecord>().get(tracker);
    GlobalPoint gp = tracker->idToDet(
        hit->geographicalId())->surface().toGlobal(hit->localPosition());
    init(gp);
  }

  SeedingHitImpl(const TransientTrackingRecHit::ConstRecHitPointer& ttrh, const SeedingLayer &layer)
    : theLayer(layer), theTransientRecHit(ttrh), theRecHit(0), hasTransientHit(true)
  {
    GlobalPoint gp = theTransientRecHit->globalPosition();
    init(gp);
  }

  void init(const GlobalPoint & gp) {
    thePhi =  gp.phi();
    theR = gp.perp();
    theZ = gp.z();
    theLayer.detLayer()->location();
    theRZ = (theLayer.detLayer()->location() == GeomDetEnumerators::barrel) ?  theZ : theR; 
    if (theLayer.hasPredefinedHitErrors()) {
      theErrorRZ = theLayer.predefinedHitErrorRZ(); 
      theErrorRPhi = theLayer.predefinedHitErrorRPhi();
    } else {
      theErrorRZ = 0.;
      theErrorRPhi = 0.;
    }
  }



  float phi() const {return thePhi;}
  float rOrZ() const { return theRZ; }
  float r() const {return theR; }
  float z() const {return theZ; }


  float errorRZ() const {
    if (!hasHitErrors) setErrorsFromTTRH(); 
    return theErrorRZ;
  }

 
  float errorRPhi() const {
    if (!hasHitErrors) setErrorsFromTTRH(); 
    return theErrorRPhi;
  }


  const TrackingRecHit * trackingRecHit() const { 
    return hasTransientHit ? theTransientRecHit->hit() : theRecHit; 
  }


  const TransientTrackingRecHit::ConstRecHitPointer & transientRecHit() const {
    if (!hasTransientHit) {
      theTransientRecHit = theLayer.hitBuilder()->build(theRecHit);  
      hasTransientHit = true; 
    }
    return theTransientRecHit; 
  }

  void setErrorsFromTTRH() const {
    const TransientTrackingRecHit::ConstRecHitPointer & hit = transientRecHit();
    GlobalPoint hitPos = hit->globalPosition();
    GlobalError hitErr = hit->globalPositionError();
    theErrorRPhi =  r()*sqrt(hitErr.phierr(hitPos)); 
    theErrorRZ = (theLayer.detLayer()->location() == GeomDetEnumerators::barrel) ? 
                  sqrt(hitErr.czz())
                : sqrt(hitErr.rerr(hitPos));
    hasHitErrors = true;
  }

private:

  SeedingLayer theLayer;
  mutable TransientTrackingRecHit::ConstRecHitPointer theTransientRecHit;
  const TrackingRecHit *theRecHit;
  float thePhi, theR, theZ, theRZ;
  mutable bool hasTransientHit;
  mutable bool hasHitErrors;
  mutable float theErrorRZ, theErrorRPhi;
};

SeedingHit::SeedingHit( const TransientTrackingRecHit::ConstRecHitPointer& ttrh, 
    const SeedingLayer& layer)
{
  SeedingHitImpl * h = new SeedingHitImpl(ttrh, layer);
  theImpl = boost::shared_ptr<SeedingHitImpl>(h);
}

SeedingHit::SeedingHit( const TrackingRecHit* hit, 
   const SeedingLayer &layer,  const edm::EventSetup& es)
{
  SeedingHitImpl * h = new SeedingHitImpl(hit,layer,es);
  theImpl = boost::shared_ptr<SeedingHitImpl>(h);
}

SeedingHit::SeedingHit( const TrackingRecHit* hit, const edm::EventSetup& es)
{
  SeedingHitImpl * h = new SeedingHitImpl(hit,es);
  theImpl = boost::shared_ptr<SeedingHitImpl>(h);
}

float SeedingHit::phi() const { return theImpl->phi(); }
float SeedingHit::rOrZ() const { return theImpl->rOrZ(); }
float SeedingHit::r() const { return theImpl->r(); }
float SeedingHit::z() const { return theImpl->z(); }
float SeedingHit::errorRZ() const { return theImpl->errorRZ(); }
float SeedingHit::errorRPhi() const { return theImpl->errorRPhi(); }

SeedingHit::operator const TrackingRecHit* () const {
  return  theImpl->trackingRecHit();
}

SeedingHit::operator const TransientTrackingRecHit::ConstRecHitPointer& () const {
  return theImpl->transientRecHit();
}

const TrackingRecHit * SeedingHit::RecHit() const { return theImpl->trackingRecHit(); }


