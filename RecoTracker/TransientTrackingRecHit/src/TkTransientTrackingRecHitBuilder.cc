#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//
// include all the concrete ones
//
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit1D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
//
// For FAMOS
//
// #include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/GSTransientTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"                         
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"                         

#include "Utilities/General/interface/ClassName.h"
#include <typeinfo>

TkTransientTrackingRecHitBuilder::TkTransientTrackingRecHitBuilder( const TrackingGeometry* trackingGeometry, 
								    const PixelClusterParameterEstimator * pCPE,
								    const StripClusterParameterEstimator * sCPE,
								    const SiStripRecHitMatcher * matcher,
								    bool computeCoarseLocalPositionFromDisk):  
  tGeometry_(trackingGeometry),
  pixelCPE(pCPE),
  stripCPE(sCPE),
  theMatcher(matcher),
  theComputeCoarseLocalPosition(computeCoarseLocalPositionFromDisk){}
  
TransientTrackingRecHit::RecHitPointer 
TkTransientTrackingRecHitBuilder::build (const TrackingRecHit * p) const 
{
  std::type_info const & tp = typeid(*p);
 
  if (tp == typeid(SiPixelRecHit))
    { 
      const SiPixelRecHit* ph = reinterpret_cast<const SiPixelRecHit*>(p); 
      return TSiPixelRecHit::build( tGeometry_->idToDet(p->geographicalId()), ph, pixelCPE, theComputeCoarseLocalPosition);
    } 
  else if (tp == typeid(SiStripRecHit2D))
    {
      const SiStripRecHit2D* sh = reinterpret_cast<const SiStripRecHit2D*>(p);
      return TSiStripRecHit2DLocalPos::build(tGeometry_->idToDet(p->geographicalId()), sh, stripCPE,  theComputeCoarseLocalPosition );
    } 
  else if (tp == typeid(SiStripRecHit1D))
    {
      const SiStripRecHit1D* sh = reinterpret_cast<const SiStripRecHit1D*>(p); 
      return TSiStripRecHit1D::build(tGeometry_->idToDet(p->geographicalId()), sh, stripCPE,  theComputeCoarseLocalPosition );
    }
  else if (tp == typeid(SiStripMatchedRecHit2D))
    {
      const SiStripMatchedRecHit2D* mh = reinterpret_cast<const SiStripMatchedRecHit2D*>(p);
      return TSiStripMatchedRecHit::build(tGeometry_->idToDet(p->geographicalId()), mh, theMatcher, stripCPE,  theComputeCoarseLocalPosition); 
    }
  else if (tp == typeid(InvalidTrackingRecHit))
    {
      return InvalidTransientRecHit::build((p->geographicalId().rawId() == 0 ? 0 : 
					    tGeometry_->idToDet(p->geographicalId())),
					   p->getType()
					   );
    }
  else if (tp == typeid(ProjectedSiStripRecHit2D))
    {
      const ProjectedSiStripRecHit2D* ph = reinterpret_cast<const ProjectedSiStripRecHit2D*>(p);
      return ProjectedRecHit2D::build(tGeometry_->idToDet(p->geographicalId()),
				      tGeometry_->idToDet(ph->originalHit().geographicalId()),
				      ph,stripCPE,
				      theComputeCoarseLocalPosition);
    }
  else if (tp == typeid(SiTrackerGSRecHit2D))
    {
      const SiTrackerGSRecHit2D* gh = reinterpret_cast<const SiTrackerGSRecHit2D*>(p);
      return ( GSTransientTrackingRecHit::build(tGeometry_->idToDet(p->geographicalId()), gh )); 
    }
  else if (tp == typeid(SiTrackerGSMatchedRecHit2D))
    {
      const SiTrackerGSMatchedRecHit2D* gh = reinterpret_cast<const SiTrackerGSMatchedRecHit2D*>(p);
      return ( GSTransientTrackingRecHit::build(tGeometry_->idToDet(p->geographicalId()), gh )); 
    } 
  return oldbuild(p);
}

TransientTrackingRecHit::RecHitPointer 
TkTransientTrackingRecHitBuilder::oldbuild (const TrackingRecHit * p) const 
{
  if ( const SiPixelRecHit* ph = dynamic_cast<const SiPixelRecHit*>(p)) {
    return ( TSiPixelRecHit::build( tGeometry_->idToDet(p->geographicalId()), ph, pixelCPE, theComputeCoarseLocalPosition) ); 
  } else if ( const SiStripRecHit2D* sh = dynamic_cast<const SiStripRecHit2D*>(p)) { 
    return ( TSiStripRecHit2DLocalPos::build(tGeometry_->idToDet(p->geographicalId()), sh, stripCPE,  theComputeCoarseLocalPosition ) ); 
  } else if ( const SiStripRecHit1D* sh = dynamic_cast<const SiStripRecHit1D*>(p)) { 
    return ( TSiStripRecHit1D::build(tGeometry_->idToDet(p->geographicalId()), sh, stripCPE,  theComputeCoarseLocalPosition ) ); 
  } else if ( const SiStripMatchedRecHit2D* mh = dynamic_cast<const SiStripMatchedRecHit2D*>(p)) {
    return ( TSiStripMatchedRecHit::build(tGeometry_->idToDet(p->geographicalId()), mh, theMatcher, stripCPE,  theComputeCoarseLocalPosition)); 
  } else if (dynamic_cast<const InvalidTrackingRecHit*>(p)){
    return ( InvalidTransientRecHit::build((p->geographicalId().rawId() == 0 ? 0 : 
					    tGeometry_->idToDet(p->geographicalId())),
					   p->getType()
					   ) );
    
  }else if (const ProjectedSiStripRecHit2D* ph = dynamic_cast<const ProjectedSiStripRecHit2D*>(p)) {
    return ProjectedRecHit2D::build(tGeometry_->idToDet(p->geographicalId()),
				    tGeometry_->idToDet(ph->originalHit().geographicalId()),
				    ph,stripCPE,
				    theComputeCoarseLocalPosition);
  } else if ( const SiTrackerGSRecHit2D* gh = dynamic_cast<const SiTrackerGSRecHit2D*>(p)) {
    return ( GSTransientTrackingRecHit::build(tGeometry_->idToDet(p->geographicalId()), gh )); 

  } else if ( const SiTrackerGSMatchedRecHit2D* gh = dynamic_cast<const SiTrackerGSMatchedRecHit2D*>(p)) {
    return ( GSTransientTrackingRecHit::build(tGeometry_->idToDet(p->geographicalId()), gh )); 
  } 
  
  throw cms::Exception("LogicError") << "TrackingRecHit* cannot be casted to a known concrete type. hit type is: "<< className(*p);
}


TransientTrackingRecHit::RecHitPointer
TkTransientTrackingRecHitBuilder::build (const TrackingRecHit * p,
					 const TrajectoryStateOnSurface & tsos) const
{
  TransientTrackingRecHit::RecHitPointer noRefit = build(p);
  return noRefit->clone(tsos);
}
