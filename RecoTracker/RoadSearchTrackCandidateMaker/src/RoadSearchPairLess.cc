#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"
#include "RecoTracker/RoadSearchTrackCandidateMaker/interface/RoadSearchPairLess.h"

#include "Utilities/General/interface/CMSexception.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool RoadSearchPairLess::InsideOutCompare(const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> HitTM1 ,
					  const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> HitTM2 ) const
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
	 float diff = HitTM1.first->globalPosition().perp() - HitTM2.first->globalPosition().perp();
	 if (std::abs(diff)<1.0e-9) return false;
	 else return (diff < 0);    
       }
       
       if( ((unsigned int)ida.subdetId() == StripSubdetector::TEC || (unsigned int)ida.subdetId() == StripSubdetector::TID || (unsigned int)ida.subdetId() == PixelSubdetector::PixelEndcap) &&
	   ((unsigned int)idb.subdetId() == StripSubdetector::TEC || (unsigned int)idb.subdetId() == StripSubdetector::TID || (unsigned int)idb.subdetId() == PixelSubdetector::PixelEndcap)) {  // fwd with fwd
	 float diff = std::abs( HitTM1.first->globalPosition().z() ) - std::abs( HitTM2.first->globalPosition().z() );
	 if (std::abs(diff)<1.0e-9) return false;
	 else return (diff < 0);
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
       throw Genexception("RoadSearchPairLess: arguments are not Ok");
       
       
     }

