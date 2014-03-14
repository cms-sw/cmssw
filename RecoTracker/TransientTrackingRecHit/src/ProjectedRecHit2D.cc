#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"


ProjectedRecHit2D::ProjectedRecHit2D( const LocalPoint& pos, const LocalError& err,
				      const GeomDet* det, const GeomDet* originalDet,
				      const TransientTrackingRecHit& originalTransientHit) :
GenericTransientTrackingRecHit( det, new ProjectedSiStripRecHit2D( pos, err, det->geographicalId(), det,
								     static_cast<const SiStripRecHit2D*>(originalTransientHit.hit()) ) ) 
{
  const TSiStripRecHit2DLocalPos* specificOriginalTransientHit = static_cast<const TSiStripRecHit2DLocalPos*>(&originalTransientHit);
  theCPE = specificOriginalTransientHit->cpe();
  theOriginalDet = originalDet;
}

ProjectedRecHit2D::RecHitPointer 
ProjectedRecHit2D::clone( const TrajectoryStateOnSurface& ts) const
{
  if (theCPE != 0) {
    TrackingRecHitProjector<ProjectedRecHit2D> proj;
    if(!originalHit().cluster().isNull()){
      const SiStripCluster& clust = *(originalHit().cluster());  
      
      const GeomDetUnit * gdu = reinterpret_cast<const GeomDetUnit *>(theOriginalDet);
      //if (!gdu) std::cout<<"no luck dude"<<std::endl;
      StripClusterParameterEstimator::LocalValues lv = 
	theCPE->localParameters( clust, *gdu, ts);
      
      RecHitPointer updatedOriginalHit = 
	TSiStripRecHit2DLocalPos::build( lv.first, lv.second, theOriginalDet, 
					 originalHit().cluster(), theCPE);
      
      RecHitPointer hit = proj.project( *updatedOriginalHit, *det(), ts); 
      
    return hit;
    }else{
      const SiStripCluster& clust = *(originalHit().cluster_regional());  
      
      const GeomDetUnit * gdu = reinterpret_cast<const GeomDetUnit *>(theOriginalDet);
      StripClusterParameterEstimator::LocalValues lv = 
	theCPE->localParameters( clust, *gdu, ts);
      
      RecHitPointer updatedOriginalHit = 
	TSiStripRecHit2DLocalPos::build( lv.first, lv.second, theOriginalDet, 
					 originalHit().cluster_regional(), theCPE);
      
      RecHitPointer hit = proj.project( *updatedOriginalHit, *det(), ts); 
      
      return hit;
      
    }
  }
  /// FIXME: should report the problem somehow
  else return clone();
}
  
TransientTrackingRecHit::ConstRecHitContainer 	
ProjectedRecHit2D::transientHits () const {
  ConstRecHitContainer result;
  result.push_back(TSiStripRecHit2DLocalPos::build( theOriginalDet,&originalHit(),theCPE));
						    
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
      if(!originalHit().cluster().isNull()){
	const SiStripCluster& clust = *(originalHit().cluster());  
	
	StripClusterParameterEstimator::LocalValues lv =
            theCPE->localParameters( clust, *reinterpret_cast<const GeomDetUnit *>(theOriginalDet));
	
	RecHitPointer updatedOriginalHit = 
	  TSiStripRecHit2DLocalPos::build( lv.first, lv.second, theOriginalDet, 
					   originalHit().cluster(), theCPE);
	
	RecHitPointer hit = proj.project( *updatedOriginalHit, *det()); 
	trackingRecHit_ = hit->hit()->clone();
      }else{
	const SiStripCluster& clust = *(originalHit().cluster_regional());  
	
	StripClusterParameterEstimator::LocalValues lv =
            theCPE->localParameters( clust, *reinterpret_cast<const GeomDetUnit *>(theOriginalDet));
	
	RecHitPointer updatedOriginalHit = 
	  TSiStripRecHit2DLocalPos::build( lv.first, lv.second, theOriginalDet, 
					   originalHit().cluster_regional(), theCPE);
	
	RecHitPointer hit = proj.project( *updatedOriginalHit, *det()); 
	trackingRecHit_ = hit->hit()->clone();
      }
    }
  }
}
