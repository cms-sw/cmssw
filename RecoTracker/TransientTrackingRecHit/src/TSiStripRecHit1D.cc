#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit1D.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"


TransientTrackingRecHit::RecHitPointer
TSiStripRecHit1D::clone (const TrajectoryStateOnSurface& ts) const
{
  if (theCPE != 0) {

    /// FIXME: this only uses the first cluster and ignores the others

     if(!specificHit()->cluster().isNull()){
       const SiStripCluster&  clust = *specificHit()->cluster();  
       StripClusterParameterEstimator::LocalValues lv = 
	 theCPE->localParameters( clust, *detUnit(), ts.localParameters());
       LocalError le(lv.second.xx(),0.,DBL_MAX); //Correct??

       return TSiStripRecHit1D::build( lv.first, le, det(), 
				       specificHit()->cluster(), theCPE, weight(), getAnnealingFactor());
     }else{
       const SiStripCluster&  clust = *specificHit()->cluster_regional();  
       StripClusterParameterEstimator::LocalValues lv = 
	 theCPE->localParameters( clust, *detUnit(), ts.localParameters());
       LocalError le(lv.second.xx(),0.,DBL_MAX); //Correct??
       return TSiStripRecHit1D::build( lv.first, le, det(), 
				       specificHit()->cluster_regional(), theCPE, weight(), getAnnealingFactor());       
     }

  }
  else {
    //FIXME. It should report the problem with a LogWarning;
    return clone();
  }
}

const GeomDetUnit* TSiStripRecHit1D::detUnit() const
{
  return static_cast<const GeomDetUnit*>(det());
}
