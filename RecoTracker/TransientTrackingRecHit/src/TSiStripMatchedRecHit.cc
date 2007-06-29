#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"


TSiStripMatchedRecHit::RecHitPointer 
TSiStripMatchedRecHit::clone( const TrajectoryStateOnSurface& ts) const
{
  if (theMatcher != 0) {
    const SiStripMatchedRecHit2D *orig = dynamic_cast<const SiStripMatchedRecHit2D *> (this->hit());
    const GeomDet *det = this->det();
    const GluedGeomDet *gdet = dynamic_cast<const GluedGeomDet *> (det);
    if ((orig == 0) || (gdet == 0)) return this->clone(); // or just die ?
    LocalVector tkDir = (ts.isValid() ? ts.localDirection() : 
			 det->surface().toLocal( det->position()-GlobalPoint(0,0,0)));
    
    if(theCPE != 0){    
      //approximation: the ts parameter on the glued surface are used on the mono
      // and stereo surface to re-evaluate cluster parameter. A further propagation 
      //is slow// and useless (?) in this case.
      StripClusterParameterEstimator::LocalValues lvMono = 
	theCPE->localParameters( *orig->monoHit()->cluster(), *gdet->monoDet(), ts.localParameters());
      StripClusterParameterEstimator::LocalValues lvStereo = 
	theCPE->localParameters( *orig->stereoHit()->cluster(), *gdet->stereoDet(), ts.localParameters());
      
      SiStripRecHit2D monoHit = SiStripRecHit2D( lvMono.first, lvMono.second,
						 gdet->monoDet()->geographicalId(),
						 orig->monoHit()->cluster());
      
      SiStripRecHit2D stereoHit = SiStripRecHit2D( lvStereo.first, lvStereo.second,
						   gdet->stereoDet()->geographicalId(),
						   orig->stereoHit()->cluster());

      const SiStripMatchedRecHit2D* better = 
	theMatcher->match(&monoHit,&stereoHit,
			  gdet,tkDir);
      
      if (better == 0) {
	//dm::LogWarning("TSiStripMatchedRecHit") << "Refitting of a matched rechit returns NULL";
	return this->clone();
      }

      RecHitPointer result = TSiStripMatchedRecHit::build( gdet, better, theMatcher,theCPE );
      delete better; //the ownership of the object is passed to the caller of the matcher
      return result;

    }else{
      const SiStripMatchedRecHit2D *better = theMatcher->match(orig,gdet,tkDir);
      if (better == 0) {
	//edm::LogWarning("TSiStripMatchedRecHit") << "Refitting of a matched rechit returns NULL";
	return this->clone();        
      }
      RecHitPointer result = TSiStripMatchedRecHit::build( gdet, better, theMatcher,theCPE );
      delete better; //the ownership of the object is passed to the caller of the matcher
      return result;
    }
  }
  return this->clone();
   
}
