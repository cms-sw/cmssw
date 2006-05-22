#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

TSiPixelRecHit::TSiPixelRecHit( const LocalPoint& pos, const LocalError& err,
				const GeomDet* det, 
				const SiPixelCluster& clust,
				const PixelClusterParameterEstimator* cpe) :
  TransientTrackingRecHit(det), theCPE(cpe)
{
  theHitData = new SiPixelRecHit( pos, err, det->geographicalId(), &clust);
}

TSiPixelRecHit* TSiPixelRecHit::clone (const TrajectoryStateOnSurface& ts) const
{
  const SiPixelCluster& clust = *specificHit()->cluster();  
  PixelClusterParameterEstimator::LocalValues lv = 
    theCPE->localParameters( clust, *detUnit(), ts.localParameters());
  return new TSiPixelRecHit( lv.first, lv.second, det(), clust, theCPE);
}

const GeomDetUnit* TSiPixelRecHit::detUnit() const
{
  return dynamic_cast<const GeomDetUnit*>(det());
}


/*
SiPixelRecHit( const LocalPoint&, const LocalError&,
		 const DetId&, 
		 const SiPixelCluster * cluster);  
*/
