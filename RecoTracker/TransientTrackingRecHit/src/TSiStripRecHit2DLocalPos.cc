#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit1D.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"


TransientTrackingRecHit::RecHitPointer
TSiStripRecHit2DLocalPos::clone (const TrajectoryStateOnSurface& ts) const
{
  if (theCPE != 0) {
    
    /// FIXME: this only uses the first cluster and ignores the others
    const SiStripCluster&  clust = specificHit()->stripCluster();  
    StripClusterParameterEstimator::LocalValues lv = 
      theCPE->localParameters( clust, *detUnit(), ts);
    return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, det(), specificHit()->omniClusterRef(), theCPE);					    
  }
  /// FIXME: should report the problem somehow
  else return clone();
}

const GeomDetUnit* TSiStripRecHit2DLocalPos::detUnit() const
{
  return static_cast<const GeomDetUnit*>(det());
}

TransientTrackingRecHit::ConstRecHitContainer 	
TSiStripRecHit2DLocalPos::transientHits () const {
  ConstRecHitContainer result;
  SiStripRecHit1D hit1d(specificHit());
  result.push_back(TSiStripRecHit1D::build( det(),&hit1d,
					    cpe()));
  
  return result;
}
