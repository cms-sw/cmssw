#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit1D.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<limits>

TSiStripRecHit1D::TSiStripRecHit1D (const GeomDet * geom, const SiStripRecHit1D* rh,
				    const StripClusterParameterEstimator* cpe,
				    bool computeCoarseLocalPosition) : 
TValidTrackingRecHit(geom), theCPE(cpe) 
{
  if (rh->hasPositionAndError() || !computeCoarseLocalPosition)
    theHitData = SiStripRecHit1D(*rh);
  else{
    const GeomDetUnit* gdu = dynamic_cast<const GeomDetUnit*>(geom);
    LogDebug("TSiStripRecHit2DLocalPos")<<"calculating coarse position/error.";
    if (gdu){
	StripClusterParameterEstimator::LocalValues lval= theCPE->localParameters(*rh->cluster(), *gdu);
	LocalError le(lval.second.xx(),0.,std::numeric_limits<float>::max()); //Correct??
	theHitData = SiStripRecHit1D(lval.first, le, *geom,rh->cluster());
    }else{
      edm::LogError("TSiStripRecHit2DLocalPos")<<" geomdet does not cast into geomdet unit. cannot create strip local parameters.";
      theHitData = SiStripRecHit1D(*rh);
    }
  }
}

TransientTrackingRecHit::RecHitPointer
TSiStripRecHit1D::clone (const TrajectoryStateOnSurface& ts) const
{
  if (theCPE != 0) {
    /// FIXME: this only uses the first cluster and ignores the others
    const SiStripCluster&  clust = specificHit()->stripCluster();  
    StripClusterParameterEstimator::LocalValues lv = 
      theCPE->localParameters( clust, *detUnit(), ts);
    LocalError le(lv.second.xx(),0.,std::numeric_limits<float>::max()); //Correct??
    return TSiStripRecHit1D::build( lv.first, le, det(), specificHit()->omniClusterRef(), theCPE);
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
