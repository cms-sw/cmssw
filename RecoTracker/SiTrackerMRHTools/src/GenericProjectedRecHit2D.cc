#include "RecoTracker/SiTrackerMRHTools/interface/GenericProjectedRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "FWCore/Utilities/interface/Exception.h"

GenericProjectedRecHit2D::GenericProjectedRecHit2D( const LocalPoint& pos, const LocalError& err,
				      		    const GeomDet* det, const GeomDet* originalDet,
					    	    const TransientTrackingRecHit::ConstRecHitPointer originalHit,
                     				    const TrackingRecHitPropagator* propagator) :
  TrackingRecHit( *det ) //, originalHit->weight(), originalHit->getAnnealingFactor()) 
{
	theOriginalDet = originalDet;
	thePropagator = propagator;
	theOriginalTransientHit = originalHit;		
	theLp = pos;
	theLe = err;
	theProjectionMatrix = originalHit->projectionMatrix();
	theDimension = originalHit->dimension();	
	//theOriginalHit = originalTransientHit.hit()->clone();
}

AlgebraicVector GenericProjectedRecHit2D::parameters() const{
	AlgebraicVector result(2);
	result[0] = theLp.x();
	result[1] = theLp.y();
	return result;
}

TransientTrackingRecHit::RecHitPointer 
GenericProjectedRecHit2D::clone( const TrajectoryStateOnSurface& ts, const TransientTrackingRecHitBuilder* builder) const
{
	return thePropagator->project<GenericProjectedRecHit2D>(theOriginalTransientHit, *det(), ts, builder); 
}
  
