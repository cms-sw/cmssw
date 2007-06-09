#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"


TSiStripRecHit2DLocalPos::
TSiStripRecHit2DLocalPos( const LocalPoint& pos, const LocalError& err,
			  const GeomDet* det, 
			  const SiStripClusterRef clust,
			  const StripClusterParameterEstimator* cpe) :
  TransientTrackingRecHit(det), theCPE(cpe)
{
  theHitData = new SiStripRecHit2D( pos, err, det->geographicalId(), clust);
}

TSiStripRecHit2DLocalPos::
TSiStripRecHit2DLocalPos( const LocalPoint& pos, const LocalError& err,
			  const GeomDet* det, 
			  const SiStripRegionalClusterRef clust,
			  const StripClusterParameterEstimator* cpe) :
  TransientTrackingRecHit(det), theCPE(cpe)
{
  theHitData = new SiStripRecHit2D( pos, err, det->geographicalId(), clust);
}


TransientTrackingRecHit::RecHitPointer
TSiStripRecHit2DLocalPos::clone (const TrajectoryStateOnSurface& ts) const
{
  if (theCPE != 0) {

    /// FIXME: this only uses the first cluster and ignores the others

     if(!specificHit()->cluster().isNull()){
       const SiStripCluster&  clust = *specificHit()->cluster();  
           StripClusterParameterEstimator::LocalValues lv = 
      theCPE->localParameters( clust, *detUnit(), ts.localParameters());
	   return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, det(), 
						   specificHit()->cluster(), theCPE);
     }else{
       const SiStripCluster&  clust = *specificHit()->cluster_regional();  
       StripClusterParameterEstimator::LocalValues lv = 
	 theCPE->localParameters( clust, *detUnit(), ts.localParameters());
       return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, det(), 
					       specificHit()->cluster(), theCPE);       
     }

  }
  /// FIXME: should report the problem somehow
  else return clone();
}

const GeomDetUnit* TSiStripRecHit2DLocalPos::detUnit() const
{
  return dynamic_cast<const GeomDetUnit*>(det());
}
