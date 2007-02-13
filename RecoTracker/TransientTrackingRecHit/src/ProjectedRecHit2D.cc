#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"


ProjectedRecHit2D::ProjectedRecHit2D( const LocalPoint& pos, const LocalError& err,
				      const GeomDet* det, const GeomDet* originalDet,
				      const TransientTrackingRecHit& originalTransientHit) :
  GenericTransientTrackingRecHit( det, new ProjectedSiStripRecHit2D( pos, err, det->geographicalId(), 
								     dynamic_cast<const SiStripRecHit2D*>(originalTransientHit.hit()))) 
{
  const TSiStripRecHit2DLocalPos* specificOriginalTransientHit = dynamic_cast<const TSiStripRecHit2DLocalPos*>(&originalTransientHit);
  theCPE = specificOriginalTransientHit->cpe();
  theOriginalDet = originalDet;
}

ProjectedRecHit2D::RecHitPointer 
ProjectedRecHit2D::clone( const TrajectoryStateOnSurface& ts) const
{
  TrackingRecHitProjector<ProjectedRecHit2D> proj;
  const SiStripCluster& clust = *(originalHit().cluster());  

  StripClusterParameterEstimator::LocalValues lv = 
    theCPE->localParameters( clust, *detUnit(), ts.localParameters());

  RecHitPointer updatedOriginalHit = 
    TSiStripRecHit2DLocalPos::build( lv.first, lv.second, theOriginalDet, 
				     originalHit().cluster(), theCPE);

  RecHitPointer hit = proj.project( *updatedOriginalHit, *det(), ts); 

  return hit;
}
