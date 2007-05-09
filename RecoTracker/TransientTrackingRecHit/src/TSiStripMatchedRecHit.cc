#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"

TSiStripMatchedRecHit::RecHitPointer 
TSiStripMatchedRecHit::clone( const TrajectoryStateOnSurface& ts) const
{
    if (theMatcher != 0) {
        const SiStripMatchedRecHit2D *orig = dynamic_cast<const SiStripMatchedRecHit2D *> (this->hit());
        const GeomDet *det = this->det();
        const GluedGeomDet *gdet = dynamic_cast<const GluedGeomDet *> (det);
        if ((orig == 0) || (gdet == 0)) return this->clone(); // or just die ?
        LocalVector tkDir = (ts.isValid() ? ts.localDirection() : det->surface().toLocal( det->position()-GlobalPoint(0,0,0)));
        const SiStripMatchedRecHit2D *better = theMatcher->match(orig,gdet,tkDir);
        if (better == 0) {
           //edm::LogWarning("TSiStripMatchedRecHit") << "Refitting of a matched rechit returns NULL";
           return this->clone();
        }
        return TSiStripMatchedRecHit::build( gdet, better, theMatcher ); 
    } else {
        return this->clone();
    } 
}
