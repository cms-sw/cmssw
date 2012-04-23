#ifndef RECOTRACKER_TRANSIENTRECHITBUILDER_H
#define RECOTRACKER_TRANSIENTRECHITBUILDER_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"

class TkTransientTrackingRecHitBuilder GCC11_FINAL : public TransientTrackingRecHitBuilder {
  
 public:
  TkTransientTrackingRecHitBuilder (const TrackingGeometry* trackingGeometry, 
				    const PixelClusterParameterEstimator * ,
				    const StripClusterParameterEstimator * ,
                                    const SiStripRecHitMatcher           *,
				    bool computeCoarseLocalPositionFromDisk);
  TransientTrackingRecHit::RecHitPointer build (const TrackingRecHit * p) const ;
  TransientTrackingRecHit::RecHitPointer build ( const TrackingRecHit * p, const TrajectoryStateOnSurface & state)  const ;

  const PixelClusterParameterEstimator * pixelClusterParameterEstimator(){return pixelCPE;}
  const StripClusterParameterEstimator * stripClusterParameterEstimator(){return stripCPE;}
  const SiStripRecHitMatcher           * siStripRecHitMatcher(){return theMatcher;}


private:
  TransientTrackingRecHit::RecHitPointer oldbuild (const TrackingRecHit * p) const ;


 private:
  const TrackingGeometry* tGeometry_;
  const PixelClusterParameterEstimator * pixelCPE;
  const StripClusterParameterEstimator * stripCPE;
  const SiStripRecHitMatcher           * theMatcher;
  bool theComputeCoarseLocalPosition;
};


#endif
