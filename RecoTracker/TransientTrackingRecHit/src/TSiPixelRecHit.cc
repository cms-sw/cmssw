#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


TSiPixelRecHit::RecHitPointer TSiPixelRecHit::clone (const TrajectoryStateOnSurface& ts) const
{
  if (theCPE == 0){
    return new TSiPixelRecHit( det(), &theHitData, 0, weight(), getAnnealingFactor(),false);
  }else{
    const SiPixelCluster& clust = *specificHit()->cluster();  
    PixelClusterParameterEstimator::LocalValues lv = 
      theCPE->localParameters( clust, *detUnit(), ts);
    return TSiPixelRecHit::build( lv.first, lv.second, det(), specificHit()->cluster(), theCPE, weight(), getAnnealingFactor());
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
			       float weight, float annealing,
			       bool computeCoarseLocalPosition) : 
  TransientTrackingRecHit(geom, *rh, weight, annealing), theCPE(cpe) 
{
  if (rh->hasPositionAndError() || !computeCoarseLocalPosition)
    theHitData = SiPixelRecHit(*rh);
  else{
    const GeomDetUnit* gdu = dynamic_cast<const GeomDetUnit*>(geom);
    if (gdu){
      PixelClusterParameterEstimator::LocalValues lval= theCPE->localParameters(*rh->cluster(), *gdu);
      LogDebug("TSiPixelRecHit")<<"calculating coarse position/error.";
      theHitData = SiPixelRecHit(lval.first, lval.second,geom->geographicalId(),rh->cluster());
    }else{
      edm::LogError("TSiPixelRecHit") << " geomdet does not cast into geomdet unit. cannot create pixel local parameters.";
      theHitData = SiPixelRecHit(*rh);
    }
  }

  // Additionally, fill the SiPixeRecHitQuality from the PixelCPE.
  const PixelCPEBase * cpeB = dynamic_cast< const PixelCPEBase* >( cpe );
  if (cpeB) {
    theHitData.setRawQualityWord( cpeB->rawQualityWord() );
    theClusterProbComputationFlag = cpeB->clusterProbComputationFlag(); 
  }
  // else {
    //  Huh?  We are using the obsolete CPEFromDetPosition???
    //  &&& Possibly complain in here.

    //  &&& In fact, a better long-term option is to simply move this interface
    //  &&&   into the ClusterParameterEstimator and make it a real interface
    //  &&&   class.
  //  }

}



// Another private constructor.  It creates the TrackingRecHit internally, 
// avoiding redundent cloning.
TSiPixelRecHit::TSiPixelRecHit( const LocalPoint& pos, const LocalError& err,
				const GeomDet* det, 
				//				const SiPixelCluster& clust,
				clusterRef clust,
				const PixelClusterParameterEstimator* cpe,
				float weight, float annealing) :
  TransientTrackingRecHit(det,weight, annealing), 
  theHitData( pos, err, det->geographicalId(), clust),
  theCPE(cpe)
{
  // Additionally, fill the SiPixeRecHitQuality from the PixelCPE.
  const PixelCPEBase * cpeB = dynamic_cast< const PixelCPEBase* >( cpe );
  if (cpeB) {
    theHitData.setRawQualityWord( cpeB->rawQualityWord() );
    theClusterProbComputationFlag = cpeB->clusterProbComputationFlag(); 
  }
  // else {
    //  Huh?  We are using the obsolete CPEFromDetPosition???
    //  &&& Possibly complain in here.

    //  &&& In fact, a better long-term option is to simply move this interface
    //  &&&   into the ClusterParameterEstimator and make it a real interface
    //  &&&   class.
  //  }
}




/*
SiPixelRecHit( const LocalPoint&, const LocalError&,
		 const DetId&, 
		 const SiPixelCluster * cluster);  
*/
