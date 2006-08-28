#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

#include "Utilities/General/interface/CMSexception.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

bool TrackingRecHitLessFromGlobalPosition::insideOutLess(  const TrackingRecHit& a, const TrackingRecHit& b) const{
  
  DetId ida(a.geographicalId());
  DetId idb(b.geographicalId());

  //(ida==idb) return false;

  if( (ida.subdetId() == StripSubdetector::TIB || ida.subdetId() == StripSubdetector::TOB || ida.subdetId() == PixelSubdetector::PixelBarrel) &&
      (idb.subdetId() == StripSubdetector::TIB || idb.subdetId() == StripSubdetector::TOB || idb.subdetId() == PixelSubdetector::PixelBarrel)) {  // barrel with barrel
    float diff = geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).perp() - geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).perp();
    if (std::abs(diff)<1.0e-9) return false;
    else return (diff < 0);    
    //return  geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).perp()< geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).perp();
  }
  
  if( (ida.subdetId() == StripSubdetector::TEC || ida.subdetId() == StripSubdetector::TID || ida.subdetId() == PixelSubdetector::PixelEndcap) &&
      (idb.subdetId() == StripSubdetector::TEC || idb.subdetId() == StripSubdetector::TID || idb.subdetId() == PixelSubdetector::PixelEndcap)) {  // fwd with fwd
    float diff = std::abs( geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).z()) - 
                 std::abs( geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).z());
    if (std::abs(diff)<1.0e-9) return false;
    else return (diff < 0);
    //return std::abs( geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).z()) < std::abs( geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).z());
  }
  
  //
  //  here I have 1 barrel against one forward
  //
  
  if( (ida.subdetId() == StripSubdetector::TIB || ida.subdetId() == StripSubdetector::TOB || ida.subdetId() == PixelSubdetector::PixelBarrel) &&
      (idb.subdetId() == StripSubdetector::TEC || idb.subdetId() == StripSubdetector::TID || idb.subdetId() == PixelSubdetector::PixelEndcap)) {  // barrel with barrel
    return barrelForwardLess( a, b);
  }else{
    return !barrelForwardLess( b, a);
  }
  
  //throw DetLogicError("GeomDetLess: arguments are not Barrel or Forward GeomDets");
  throw Genexception("TrackingRecHitLessFromGlobalPosition: arguments are not Ok");
}

bool TrackingRecHitLessFromGlobalPosition::barrelForwardLess(  const TrackingRecHit& a, const TrackingRecHit& b) const{
  //
  // for the moment sort again in z, but since the z in the barrel is wrong (it is in the centre of the module)
  // add the semi length
  //
  DetId ida(a.geographicalId());
  DetId idb(b.geographicalId());
  BoundPlane s =  geometry->idToDet(ida)->specificSurface();
  const Bounds * bb     = &(s.bounds());

  float diff = std::abs( geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).z())+ std::abs(bb->length()/2.) -
    std::abs( geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).z());
  if (std::abs(diff)<1.0e-9) return false;
  else return (diff < 0);
  //return  std::abs( geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).z())+ std::abs(bb->length()/2.) < std::abs( geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).z());
}
