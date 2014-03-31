#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"


ProjectedRecHit2D::ProjectedRecHit2D( const LocalPoint& pos, const LocalError& err,
				      const GeomDet* det, const GeomDet* originalDet,
				      const TransientTrackingRecHit& originalTransientHit) :
GenericTransientTrackingRecHit( det, new ProjectedSiStripRecHit2D( pos, err, *det,
								     *static_cast<const SiStripRecHit2D*>(originalTransientHit.hit()) ) ) 
{
  const TSiStripRecHit2DLocalPos* specificOriginalTransientHit = static_cast<const TSiStripRecHit2DLocalPos*>(&originalTransientHit);
  theCPE = specificOriginalTransientHit->cpe();
  theOriginalDet = originalDet;
}

ProjectedRecHit2D::ProjectedRecHit2D( const LocalPoint& pos, const LocalError& err,
				      const GeomDet* det, const GeomDet* originalDet,
				      const TrackingRecHit& originalHit, const StripClusterParameterEstimator* cpe) :
GenericTransientTrackingRecHit( det, new ProjectedSiStripRecHit2D( pos, err, *det,
								     *static_cast<const SiStripRecHit2D*>(&originalHit) ) ) 
{
  theCPE = cpe;
  theOriginalDet = originalDet;
}


ProjectedRecHit2D::RecHitPointer 
ProjectedRecHit2D::clone( const TrajectoryStateOnSurface& ts) const
{
  if (theCPE != 0) {
    TrackingRecHitProjector<ProjectedRecHit2D> proj;
      const SiStripCluster& clust = *(specificHit().cluster());  
      
      const GeomDetUnit * gdu = reinterpret_cast<const GeomDetUnit *>(theOriginalDet);
      //if (!gdu) std::cout<<"no luck dude"<<std::endl;
      StripClusterParameterEstimator::LocalValues lv = 
	theCPE->localParameters( clust, *gdu, ts);
      
      RecHitPointer updatedOriginalHit = 
	TSiStripRecHit2DLocalPos::build( lv.first, lv.second, theOriginalDet, 
					 specificHit().cluster(), theCPE);
      
      RecHitPointer hit = proj.project( *updatedOriginalHit, *det(), ts); 
      
    return hit;
  }
  /// FIXME: should report the problem somehow
  else return clone();
}
  
TransientTrackingRecHit::ConstRecHitContainer 	
ProjectedRecHit2D::transientHits () const {
  ConstRecHitContainer result;
  const SiStripCluster& clust = specificHit().stripCluster();

  const GeomDetUnit * gdu = reinterpret_cast<const GeomDetUnit *>(theOriginalDet);
  //if (!gdu) std::cout<<"no luck dude"<<std::endl;
  StripClusterParameterEstimator::LocalValues lv =
        theCPE->localParameters( clust, *gdu);

  result.push_back(TSiStripRecHit2DLocalPos::build(lv.first, lv.second, theOriginalDet,
                                                   specificHit().omniCluster(), theCPE));
						    
  return result;
}

ProjectedRecHit2D::ProjectedRecHit2D( const GeomDet * geom, const GeomDet* originaldet,
				      const ProjectedSiStripRecHit2D* rh,
				      const StripClusterParameterEstimator* cpe,
				      bool computeCoarseLocalPosition) :
  GenericTransientTrackingRecHit( geom, *rh), theCPE(cpe), theOriginalDet(originaldet) {
  if (computeCoarseLocalPosition){
    if (theCPE != 0) {
      TrackingRecHitProjector<ProjectedRecHit2D> proj;
	const SiStripCluster& clust = *(specificHit().cluster());  
	
	StripClusterParameterEstimator::LocalValues lv =
            theCPE->localParameters( clust, *reinterpret_cast<const GeomDetUnit *>(theOriginalDet));
	
	RecHitPointer updatedOriginalHit = 
	  TSiStripRecHit2DLocalPos::build( lv.first, lv.second, theOriginalDet, 
					   specificHit().cluster(), theCPE);
	
	RecHitPointer hit = proj.project( *updatedOriginalHit, *det()); 
	trackingRecHit_ = hit->hit()->clone();
    }
  }
}
