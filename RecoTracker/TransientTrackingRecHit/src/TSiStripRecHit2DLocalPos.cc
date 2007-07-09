#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"


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
  return static_cast<const GeomDetUnit*>(det());
}
