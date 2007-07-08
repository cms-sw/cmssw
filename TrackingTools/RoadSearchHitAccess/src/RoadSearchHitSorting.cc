//
// Package:         TrackingTools/RoadSearchHitSorting
// Class:           SortHitsByGlobalPosition
//                  SortHitPointersByGLobalPosition
// 
// Description:     various sortings for TrackingRecHits
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Fri Jul  6 13:37:38 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/06/29 23:54:04 $
// $Revision: 1.39 $
//

#include "TrackingTools/RoadSearchHitAccess/interface/RoadSearchHitSorting.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Utilities/General/interface/CMSexception.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

bool SortHitsByGlobalPosition::insideOutLess(  const TrackingRecHit& a, const TrackingRecHit& b) const{
  
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
  
  throw Genexception("SortHitsByGlobalPosition: arguments are not Ok");
}

bool SortHitsByGlobalPosition::barrelForwardLess(  const TrackingRecHit& a, const TrackingRecHit& b) const{
  //
  // for the moment sort again in z, but since the z in the barrel is wrong (it is in the centre of the module)
  // add the semi length
  //
  DetId ida(a.geographicalId());
  DetId idb(b.geographicalId());
  return  static_cast<unsigned int>(std::abs( geometry->idToDet(ida)->surface().toGlobal(a.localPosition()).z()) * 1E7) < static_cast<unsigned int>(std::abs( geometry->idToDet(idb)->surface().toGlobal(b.localPosition()).z()) * 1E7);
}

bool SortHitPointersByGlobalPosition::insideOutLess(  const TrackingRecHit* a, const TrackingRecHit* b) const{
  
  DetId ida(a->geographicalId());
  DetId idb(b->geographicalId());

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
    return  static_cast<unsigned int>(geometry->idToDet(ida)->surface().toGlobal(a->localPosition()).perp() * 1E7) < static_cast<unsigned int>(geometry->idToDet(idb)->surface().toGlobal(b->localPosition()).perp() * 1E7);
  }
  
  if( (idetA == StripSubdetector::TEC || idetA == StripSubdetector::TID || idetA == PixelSubdetector::PixelEndcap) &&
      (idetB == StripSubdetector::TEC || idetB == StripSubdetector::TID || idetB == PixelSubdetector::PixelEndcap)) {  
    return static_cast<unsigned int>(std::abs(geometry->idToDet(ida)->surface().toGlobal(a->localPosition()).z()) * 1E7) < static_cast<unsigned int>(std::abs(geometry->idToDet(idb)->surface().toGlobal(b->localPosition()).z()) * 1E7);
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
  
  throw Genexception("SortHitPointersByGlobalPosition: arguments are not Ok");
}

bool SortHitPointersByGlobalPosition::barrelForwardLess(  const TrackingRecHit* a, const TrackingRecHit* b) const{
  //
  // for the moment sort again in z, but since the z in the barrel is wrong (it is in the centre of the module)
  // add the semi length
  //
  DetId ida(a->geographicalId());
  DetId idb(b->geographicalId());
  return  static_cast<unsigned int>(std::abs( geometry->idToDet(ida)->surface().toGlobal(a->localPosition()).z()) * 1E7) < static_cast<unsigned int>(std::abs( geometry->idToDet(idb)->surface().toGlobal(b->localPosition()).z()) * 1E7);
}

bool SortHitTrajectoryPairsByGlobalPosition::InsideOutCompare(const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*>& HitTM1 ,
							      const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*>& HitTM2 ) const
{


  DetId ida(HitTM1.first->det()->geographicalId());
  DetId idb(HitTM2.first->det()->geographicalId());

       
  LogDebug("RoadSearch")<<" Comparing (r/phi/z) Hit 1 on DetID "
			<< ida.rawId() << " : "
			<< HitTM1.first->globalPosition().perp() << " / "
			<< HitTM1.first->globalPosition().phi() << " / "
			<< HitTM1.first->globalPosition().z()
			<< " and Hit 2 on DetID "
			<< idb.rawId() << " : "
			<< HitTM2.first->globalPosition().perp() << " / "
			<< HitTM2.first->globalPosition().phi() << " / "
			<< HitTM2.first->globalPosition().z() ;
       

  if( ((unsigned int)ida.subdetId() == StripSubdetector::TIB || (unsigned int)ida.subdetId() == StripSubdetector::TOB || (unsigned int)ida.subdetId() == PixelSubdetector::PixelBarrel) &&
      ((unsigned int)idb.subdetId() == StripSubdetector::TIB || (unsigned int)idb.subdetId() == StripSubdetector::TOB || (unsigned int)idb.subdetId() == PixelSubdetector::PixelBarrel)) {  // barrel with barrel
    return  static_cast<unsigned int>(HitTM1.first->globalPosition().perp() * 1E7) < static_cast<unsigned int>(HitTM2.first->globalPosition().perp() * 1E7);
  }
       
  if( ((unsigned int)ida.subdetId() == StripSubdetector::TEC || (unsigned int)ida.subdetId() == StripSubdetector::TID || (unsigned int)ida.subdetId() == PixelSubdetector::PixelEndcap) &&
      ((unsigned int)idb.subdetId() == StripSubdetector::TEC || (unsigned int)idb.subdetId() == StripSubdetector::TID || (unsigned int)idb.subdetId() == PixelSubdetector::PixelEndcap)) {  // fwd with fwd
    return  static_cast<unsigned int>(HitTM1.first->globalPosition().z() * 1E7) < static_cast<unsigned int>(HitTM2.first->globalPosition().z() * 1E7);
  }
       
  //
  //  here I have 1 barrel against one forward
  //
       
  if( ((unsigned int)ida.subdetId() == StripSubdetector::TIB || (unsigned int)ida.subdetId() == StripSubdetector::TOB || (unsigned int)ida.subdetId() == PixelSubdetector::PixelBarrel) &&
      ((unsigned int)idb.subdetId() == StripSubdetector::TEC || (unsigned int)idb.subdetId() == StripSubdetector::TID || (unsigned int)idb.subdetId() == PixelSubdetector::PixelEndcap)) {  // barrel with barrel
    LogDebug("RoadSearch") <<"*** How did this happen ?!?!? ***" ;
  }else{
    LogDebug("RoadSearch") <<"*** How did this happen ?!?!? ***" ;
  }
       
  //throw DetLogicError("GeomDetLess: arguments are not Barrel or Forward GeomDets");
  throw Genexception("SortHitTrajectoryPairsByGlobalPosition: arguments are not Ok");
       
       
}

