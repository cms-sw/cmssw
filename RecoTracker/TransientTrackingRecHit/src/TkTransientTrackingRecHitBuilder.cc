#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//
// include all the concrete ones
//
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"


TkTransientTrackingRecHitBuilder::TkTransientTrackingRecHitBuilder( const TrackingGeometry* trackingGeometry, 
								    const PixelClusterParameterEstimator * p,
								    const StripClusterParameterEstimator * s):  
  tGeometry_(trackingGeometry),
  pixelCPE(p),
  stripCPE(s) {}
  
TransientTrackingRecHit::RecHitPointer 
TkTransientTrackingRecHitBuilder::build (const TrackingRecHit * p) const 
{
  if ( const SiStripRecHit2D* sh = dynamic_cast<const SiStripRecHit2D*>(p)) { 
    return ( TSiStripRecHit2DLocalPos::build(tGeometry_->idToDet(p->geographicalId()), sh, stripCPE ) ); 
  } else if ( const SiStripMatchedRecHit2D* mh = dynamic_cast<const SiStripMatchedRecHit2D*>(p)) {
    return ( TSiStripMatchedRecHit::build(tGeometry_->idToDet(p->geographicalId()), mh )); 
  } else if ( const SiPixelRecHit* ph = dynamic_cast<const SiPixelRecHit*>(p)) {
    return ( TSiPixelRecHit::build( tGeometry_->idToDet(p->geographicalId()), ph, pixelCPE) ); 
  }else if (dynamic_cast<const InvalidTrackingRecHit*>(p)){
    return ( InvalidTransientRecHit::build( (p->geographicalId().rawId() == 0 ? 0 : 
					     tGeometry_->idToDet(p->geographicalId())
					     ) ));  
  }else if (const ProjectedSiStripRecHit2D* ph = dynamic_cast<const ProjectedSiStripRecHit2D*>(p)) {
    return ProjectedRecHit2D::build(tGeometry_->idToDet(p->geographicalId()),ph,stripCPE);
  } 
  throw cms::Exception("LogicError") << "TrackingRecHit* cannot be casted to a known concrete type"; 
}
