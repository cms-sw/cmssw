#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include<typeinfo>

TSiPixelRecHit::RecHitPointer TSiPixelRecHit::clone (const TrajectoryStateOnSurface& ts) const
{
  if (theCPE == 0){
    return new TSiPixelRecHit( det(), &theHitData, 0,false);
  }else{
    const SiPixelCluster& clust = *specificHit()->cluster();  
    PixelClusterParameterEstimator::LocalValues lv = 
      theCPE->localParameters( clust, *detUnit(), ts);
    return TSiPixelRecHit::build( lv.first, lv.second, theCPE->rawQualityWord(), det(), specificHit()->cluster(), theCPE);
  }
}

const GeomDetUnit* TSiPixelRecHit::detUnit() const
{
  return static_cast<const GeomDetUnit*>(det());
}



// This private constructor copies the TrackingRecHit.  It should be used when the 
// TrackingRecHit exist already in some collection.
TSiPixelRecHit::TSiPixelRecHit(const GeomDet * geom, const SiPixelRecHit* rh, 
			       const PixelClusterParameterEstimator* cpe,
			       bool computeCoarseLocalPosition) : 
  TValidTrackingRecHit(geom, *rh), theCPE(cpe), theHitData(*rh)
{
  if (! (rh->hasPositionAndError() || !computeCoarseLocalPosition)) {
    const GeomDetUnit* gdu = dynamic_cast<const GeomDetUnit*>(geom);
    if (gdu){
      PixelClusterParameterEstimator::LocalValues lval= theCPE->localParameters(*rh->cluster(), *gdu);
      LogDebug("TSiPixelRecHit")<<"calculating coarse position/error.";
      theHitData = SiPixelRecHit(lval.first, lval.second, theCPE->rawQualityWord(), *geom, rh->cluster());
    }else{
      edm::LogError("TSiPixelRecHit") << " geomdet does not cast into geomdet unit. cannot create pixel local parameters.";
    }
  }
}
