#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

TSiPixelRecHit::TSiPixelRecHit( const LocalPoint& pos, const LocalError& err,
				const GeomDet* det, 
				//				const SiPixelCluster& clust,
				clusterRef clust,
				const PixelClusterParameterEstimator* cpe) :
  TransientTrackingRecHit(det), theCPE(cpe)
{
  theHitData = new SiPixelRecHit( pos, err, det->geographicalId(), clust);
}

TSiPixelRecHit::RecHitPointer TSiPixelRecHit::clone (const TrajectoryStateOnSurface& ts) const
{
  if (theCPE == 0){
    return new TSiPixelRecHit( det(), theHitData, 0);
  }else{
    const SiPixelCluster& clust = *specificHit()->cluster();  
    PixelClusterParameterEstimator::LocalValues lv = 
      theCPE->localParameters( clust, *detUnit(), ts.localParameters());
    return TSiPixelRecHit::build( lv.first, lv.second, det(), specificHit()->cluster(), theCPE);
  }
}

const GeomDetUnit* TSiPixelRecHit::detUnit() const
{
  return static_cast<const GeomDetUnit*>(det());
}


/*
SiPixelRecHit( const LocalPoint&, const LocalError&,
		 const DetId&, 
		 const SiPixelCluster * cluster);  
*/
