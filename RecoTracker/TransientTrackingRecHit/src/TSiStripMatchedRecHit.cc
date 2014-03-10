#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"


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

TSiStripMatchedRecHit::RecHitPointer 
TSiStripMatchedRecHit::clone( const TrajectoryStateOnSurface& ts) const
{
  if likely( (theMatcher != nullptr) & (theCPE != nullptr)) {
    const SiStripMatchedRecHit2D *orig = static_cast<const SiStripMatchedRecHit2D *> (this->hit());
    const GeomDet *det = this->det();
    const GluedGeomDet *gdet = static_cast<const GluedGeomDet *> (det);
    //if ((orig == 0) || (gdet == 0)) return this->clone(); // or just die ?
    LocalVector tkDir = (ts.isValid() ? ts.localDirection() : 
			 det->surface().toLocal( det->position()-GlobalPoint(0,0,0)));
    
      //approximation: the ts parameter on the glued surface are used on the mono
      // and stereo surface to re-evaluate cluster parameter. A further propagation 
      //is slow// and useless (?) in this case.

    const SiStripCluster& monoclust   = orig->monoCluster();  
    const SiStripCluster& stereoclust = orig->stereoCluster();
    
    StripClusterParameterEstimator::LocalValues lvMono = 
      theCPE->localParameters( monoclust, *gdet->monoDet(), ts);
    StripClusterParameterEstimator::LocalValues lvStereo = 
      theCPE->localParameters( stereoclust, *gdet->stereoDet(), gluedToStereo(ts, gdet));
      
    SiStripRecHit2D monoHit = SiStripRecHit2D( lvMono.first, lvMono.second, 
					       *gdet->monoDet(),
					       orig->monoClusterRef());
    SiStripRecHit2D stereoHit = SiStripRecHit2D( lvStereo.first, lvStereo.second,
						 *gdet->stereoDet(),
						 orig->stereoClusterRef());
    const SiStripMatchedRecHit2D* better =  theMatcher->match(&monoHit,&stereoHit,gdet,tkDir);
    
    if (better == nullptr) {
      //dm::LogWarning("TSiStripMatchedRecHit") << "Refitting of a matched rechit returns NULL";
      return this->clone();
    }
    
    return RecHitPointer(new TSiStripMatchedRecHit( gdet, better, theMatcher,theCPE, false, DontCloneRecHit()));
    // delete better; //the ownership of the object is passed to the caller of the matcher
    
  }
  return this->clone();
   
}



TransientTrackingRecHit::ConstRecHitContainer 	
TSiStripMatchedRecHit::transientHits () const {
  ConstRecHitContainer result;

  const GluedGeomDet *gdet = static_cast<const GluedGeomDet *> (this->det());
  const SiStripMatchedRecHit2D *orig = static_cast<const SiStripMatchedRecHit2D *> (this->hit());

  if (theCPE!=nullptr) {
    // this is at least the third place I read (write) this logic...
    const SiStripCluster& monoclust   = orig->monoCluster();  
    const SiStripCluster& stereoclust = orig->stereoCluster();
    
    StripClusterParameterEstimator::LocalValues lvMono = 
      theCPE->localParameters( monoclust, *gdet->monoDet());
    StripClusterParameterEstimator::LocalValues lvStereo = 
      theCPE->localParameters( stereoclust, *gdet->stereoDet());
    
    result.push_back(TSiStripRecHit2DLocalPos::build(lvMono.first, lvMono.second, gdet->monoDet(), 
						     orig->monoClusterRef(), theCPE));
    result.push_back(TSiStripRecHit2DLocalPos::build(lvStereo.first, lvStereo.second, gdet->stereoDet(), 
						     orig->stereoClusterRef(), theCPE));
  }
  else {
    auto m = orig->monoHit(); auto s = orig->stereoHit();
    result.push_back(TSiStripRecHit2DLocalPos::build( gdet->monoDet(),&m,theCPE));
    result.push_back(TSiStripRecHit2DLocalPos::build( gdet->stereoDet(),&s,theCPE));
  }
  return result;
}

  void TSiStripMatchedRecHit::ComputeCoarseLocalPosition(){
  if (!theCPE || !theMatcher) return;
  const SiStripMatchedRecHit2D *orig = static_cast<const SiStripMatchedRecHit2D *> (trackingRecHit_);
  if ( (!orig)  ||  orig->hasPositionAndError()) return;

  LogDebug("TSiStripMatchedRecHit")<<"calculating coarse position/error.";
  const GeomDet *det = this->det();
  const GluedGeomDet *gdet = static_cast<const GluedGeomDet *> (det);
  LocalVector tkDir = det->surface().toLocal( det->position()-GlobalPoint(0,0,0));
  
  const SiStripCluster& monoclust   = orig->monoCluster();  
  const SiStripCluster& stereoclust = orig->stereoCluster();
  
  StripClusterParameterEstimator::LocalValues lvMono = 
    theCPE->localParameters( monoclust, *gdet->monoDet());
  StripClusterParameterEstimator::LocalValues lvStereo = 
    theCPE->localParameters( stereoclust, *gdet->stereoDet());
 
  SiStripRecHit2D monoHit = SiStripRecHit2D( lvMono.first, lvMono.second, 
					     *gdet->monoDet(),
					     orig->monoClusterRef());
  SiStripRecHit2D stereoHit = SiStripRecHit2D( lvStereo.first, lvStereo.second, 
					       *gdet->stereoDet(),
						 orig->stereoClusterRef());
  SiStripMatchedRecHit2D* better =  theMatcher->match(&monoHit,&stereoHit,gdet,tkDir);
  
  if (!better) {
    edm::LogWarning("TSiStripMatchedRecHit")<<"could not get a matching rechit.";
  }else{
    delete trackingRecHit_;
    trackingRecHit_ = better;
  }
  
}
