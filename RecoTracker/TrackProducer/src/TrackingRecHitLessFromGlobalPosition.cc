#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

bool TrackingRecHitLessFromGlobalPosition::insideOutLess(  const TrackingRecHit& a, const TrackingRecHit& b) const{
  
  DetId ida(a.geographicalId());
  DetId idb(b.geographicalId());

  unsigned int idetA = static_cast<unsigned int>(ida.subdetId());
  unsigned int idetB = static_cast<unsigned int>(idb.subdetId());

  //check for mixed case...
  bool same_det = ( 
		   (idetA == StripSubdetector::TIB && idetB == StripSubdetector::TIB) ||
		   (idetA == StripSubdetector::TID && idetB == StripSubdetector::TID) ||
		   (idetA == StripSubdetector::TIB && idetB == StripSubdetector::TID) ||
		   (idetA == StripSubdetector::TID && idetB == StripSubdetector::TIB) ||

		   (idetA == StripSubdetector::TOB && idetB == StripSubdetector::TOB) ||
		   (idetA == StripSubdetector::TEC && idetB == StripSubdetector::TEC) ||
		   (idetA == StripSubdetector::TOB && idetB == StripSubdetector::TEC) ||
		   (idetA == StripSubdetector::TEC && idetB == StripSubdetector::TOB) ||

		   (idetA == PixelSubdetector::PixelBarrel && idetB == PixelSubdetector::PixelBarrel) ||
		   (idetA == PixelSubdetector::PixelEndcap && idetB == PixelSubdetector::PixelEndcap) ||
		   (idetA == PixelSubdetector::PixelBarrel && idetB == PixelSubdetector::PixelEndcap) ||
		   (idetA == PixelSubdetector::PixelEndcap && idetB == PixelSubdetector::PixelBarrel) );

  if (!same_det) return (idetA < idetB);

  if( (idetA == StripSubdetector::TIB || idetA == StripSubdetector::TOB || idetA == PixelSubdetector::PixelBarrel) &&
      (idetB == StripSubdetector::TIB || idetB == StripSubdetector::TOB || idetB == PixelSubdetector::PixelBarrel)) {  
    return  static_cast<unsigned int>(geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).perp() * 1E7) < static_cast<unsigned int>(geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).perp() * 1E7);
  }
  
  if( (idetA == StripSubdetector::TEC || idetA == StripSubdetector::TID || idetA == PixelSubdetector::PixelEndcap) &&
      (idetB == StripSubdetector::TEC || idetB == StripSubdetector::TID || idetB == PixelSubdetector::PixelEndcap)) {  
    return static_cast<unsigned int>(std::abs(geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).z()) * 1E7) < static_cast<unsigned int>(std::abs(geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).z()) * 1E7);
  }
  
  //
  //  here I have 1 barrel against one forward
  //
  
  if( (idetA == StripSubdetector::TIB || idetA == StripSubdetector::TOB || idetA == PixelSubdetector::PixelBarrel) &&
      (idetB == StripSubdetector::TEC || idetB == StripSubdetector::TID || idetB == PixelSubdetector::PixelEndcap)) {
    return barrelForwardLess( a, b);
  }else{
    return !barrelForwardLess( b, a);
  }
  
  throw cms::Exception("TrackingRecHitLessFromGlobalPosition", "Arguments are not Ok");
}

bool TrackingRecHitLessFromGlobalPosition::barrelForwardLess(  const TrackingRecHit& a, const TrackingRecHit& b) const{
  //
  // for the moment sort again in z, but since the z in the barrel is wrong (it is in the centre of the module)
  // add the semi length
  //
  DetId ida(a.geographicalId());
  DetId idb(b.geographicalId());
  return  static_cast<unsigned int>(std::abs( geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).z()) * 1E7) < static_cast<unsigned int>(std::abs( geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).z()) * 1E7);
}
