#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"

ProjectedRecHit2D::ProjectedRecHit2D( const LocalPoint& pos, const LocalError& err, const GeomDet* det,
				      const TransientTrackingRecHit& originalHit) :
  //  TransientTrackingRecHit(det)
  GenericTransientTrackingRecHit( det, new ProjectedSiStripRecHit2D( pos, err, det->geographicalId(), 
								     dynamic_cast<const SiStripRecHit2D*>(originalHit.hit()))), 
  theOriginalHit( &originalHit)
{
  //const TrackingRecHit& thit = *originalHit.hit();
  //const SiStripRecHit2D& siHit = dynamic_cast<const SiStripRecHit2D&>(thit);
  //theHitData = new ProjectedSiStripRecHit2D( pos, err, det->geographicalId(), siHit); 
}

ProjectedRecHit2D::RecHitPointer 
ProjectedRecHit2D::clone( const TrajectoryStateOnSurface& ts) const
{
  TrackingRecHitProjector<ProjectedRecHit2D> proj;
  //RecHitPointer updatedOriginalHit = originalHit().clone( ts); //TO FIX
  //const ProjectedSiStripRecHit2D*  updatedOriginalHit = originalHit()->clone();

  //RecHitPointer hit = proj.project( *updatedOriginalHit, *det(), ts); //TO FIX
  //return hit;
  RecHitPointer hit = build(this->det(),dynamic_cast<const ProjectedSiStripRecHit2D*>(this->hit()));
  return hit;
}
