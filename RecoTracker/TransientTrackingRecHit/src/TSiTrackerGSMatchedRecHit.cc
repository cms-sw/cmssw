#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerGSMatchedRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerGSRecHit.h"

#undef RecoTracker_TransientTrackingRecHit_TSiStripMatchedRecHit_RefitProj
#undef RecoTracker_TransientTrackingRecHit_TSiStripMatchedRecHit_RefitLGL
#ifdef RecoTracker_TransientTrackingRecHit_TSiStripMatchedRecHit_RefitLGL 
// Local lo Global lo Local
inline LocalTrajectoryParameters gluedToStereo(const TrajectoryStateOnSurface &tsos, const GluedGeomDet *gdet) {    
    const BoundPlane &stripPlane = gdet->stereoDet()->surface();
    LocalPoint  lp = stripPlane.toLocal(tsos.globalPosition());
    LocalVector ld = stripPlane.toLocal(tsos.globalParameters().momentum());
    return LocalTrajectoryParameters(lp,ld,tsos.charge());
}
#elif defined(RecoTracker_TransientTrackingRecHit_TSiStripMatchedRecHit_RefitProj)
// A la RecHitProjector
inline LocalTrajectoryParameters gluedToStereo(const TrajectoryStateOnSurface &tsos, const GluedGeomDet *gdet) {
    const BoundPlane &stripPlane = gdet->stereoDet()->surface();
    double delta = stripPlane.localZ( tsos.globalPosition());
    LocalVector ld = stripPlane.toLocal(tsos.globalParameters().momentum());
    LocalPoint  lp = stripPlane.toLocal(tsos.globalPosition()) - ld*delta/ld.z();
    return LocalTrajectoryParameters(lp,ld,tsos.charge());
}
#else
// Dummy
inline const LocalTrajectoryParameters & gluedToStereo(const TrajectoryStateOnSurface &tsos, const GluedGeomDet *gdet) {
    return tsos.localParameters();
}
#endif

const GeomDetUnit* TSiTrackerGSMatchedRecHit::detUnit() const
{
  return static_cast<const GeomDetUnit*>(det());
}

TSiTrackerGSMatchedRecHit::RecHitPointer 
TSiTrackerGSMatchedRecHit::clone( const TrajectoryStateOnSurface& ts) const
{

  return this->clone();
   
}

TransientTrackingRecHit::ConstRecHitContainer 	
TSiTrackerGSMatchedRecHit::transientHits () const {
  ConstRecHitContainer result;

  const GluedGeomDet *gdet = static_cast<const GluedGeomDet *> (this->det());
  const SiTrackerGSMatchedRecHit2D *orig = static_cast<const SiTrackerGSMatchedRecHit2D *> (this->hit());

  result.push_back(TSiTrackerGSRecHit::build( gdet->monoDet(),orig->monoHit()/*,theCPE*/));
  result.push_back(TSiTrackerGSRecHit::build( gdet->stereoDet(),orig->stereoHit()/*,theCPE*/));
  return result;
}
