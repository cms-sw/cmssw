#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"
#include "RecoTracker/RoadSearchTrackCandidateMaker/interface/RoadSearchPairLess.h"

#include "Utilities/General/interface/CMSexception.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

bool RoadSearchPairLess::InsideOutCompare(const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> HitTM1 ,
					  const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> HitTM2 ) const
     {


       DetId ida(HitTM1.first->det()->geographicalId());
       DetId idb(HitTM2.first->det()->geographicalId());

       
       std::cout<<" Comparing (r/phi/z) Hit 1 on DetID "
		<< ida.rawId() << " : "
		<< HitTM1.first->globalPosition().perp() << " / "
		<< HitTM1.first->globalPosition().phi() << " / "
		<< HitTM1.first->globalPosition().z()
		<< " and Hit 2 on DetID "
		<< idb.rawId() << " : "
		<< HitTM2.first->globalPosition().perp() << " / "
		<< HitTM2.first->globalPosition().phi() << " / "
		<< HitTM2.first->globalPosition().z() << std::endl;
       

       if( (ida.subdetId() == StripSubdetector::TIB || ida.subdetId() == StripSubdetector::TOB || ida.subdetId() == PixelSubdetector::PixelBarrel) &&
	   (idb.subdetId() == StripSubdetector::TIB || idb.subdetId() == StripSubdetector::TOB || idb.subdetId() == PixelSubdetector::PixelBarrel)) {  // barrel with barrel
	 float diff = HitTM1.first->globalPosition().perp() - HitTM2.first->globalPosition().perp();
	 if (std::abs(diff)<1.0e-9) return false;
	 else return (diff < 0);    
       }
       
       if( (ida.subdetId() == StripSubdetector::TEC || ida.subdetId() == StripSubdetector::TID || ida.subdetId() == PixelSubdetector::PixelEndcap) &&
	   (idb.subdetId() == StripSubdetector::TEC || idb.subdetId() == StripSubdetector::TID || idb.subdetId() == PixelSubdetector::PixelEndcap)) {  // fwd with fwd
	 float diff = std::abs( HitTM1.first->globalPosition().z() ) - std::abs( HitTM2.first->globalPosition().z() );
	 if (std::abs(diff)<1.0e-9) return false;
	 else return (diff < 0);
       }
       
       //
       //  here I have 1 barrel against one forward
       //
       
       if( (ida.subdetId() == StripSubdetector::TIB || ida.subdetId() == StripSubdetector::TOB || ida.subdetId() == PixelSubdetector::PixelBarrel) &&
	   (idb.subdetId() == StripSubdetector::TEC || idb.subdetId() == StripSubdetector::TID || idb.subdetId() == PixelSubdetector::PixelEndcap)) {  // barrel with barrel
	 std::cout<<"*** How did this happen ?!?!? ***" << std::endl;
       }else{
	 std::cout<<"*** How did this happen ?!?!? ***" << std::endl;
       }
       
       //throw DetLogicError("GeomDetLess: arguments are not Barrel or Forward GeomDets");
       throw Genexception("RoadSearchPairLess: arguments are not Ok");
       
       
     }

