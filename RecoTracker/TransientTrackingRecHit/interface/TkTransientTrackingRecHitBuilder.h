#ifndef RECOTRACKER_TRANSIENTRECHITBUILDER_H
#define RECOTRACKER_TRANSIENTRECHITBUILDER_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"



class SiStripRecHitMatcher;
class PixelClusterParameterEstimator;
class StripClusterParameterEstimator;

class TkTransientTrackingRecHitBuilder GCC11_FINAL : public TransientTrackingRecHitBuilder {
  
 public:
  TkTransientTrackingRecHitBuilder (const TrackingGeometry* trackingGeometry, 
				    const PixelClusterParameterEstimator * ,
				    const StripClusterParameterEstimator * ,
                                    const SiStripRecHitMatcher           *,
				    bool computeCoarseLocalPositionFromDisk);

  TransientTrackingRecHit::RecHitPointer build (const TrackingRecHit * p) const ;


  const PixelClusterParameterEstimator * pixelClusterParameterEstimator() const {return pixelCPE;}
  const StripClusterParameterEstimator * stripClusterParameterEstimator() const {return stripCPE;}
  const SiStripRecHitMatcher           * siStripRecHitMatcher() const {return theMatcher;}
  const TrackingGeometry               * geometry() const  { return tGeometry_;}

  // for the time being here...
  TkClonerImpl cloner() const { return TkClonerImpl(pixelCPE,stripCPE,theMatcher);}

private:


 private:
  const TrackingGeometry* tGeometry_;
  const PixelClusterParameterEstimator * pixelCPE;
  const StripClusterParameterEstimator * stripCPE;
  const SiStripRecHitMatcher           * theMatcher;
  bool theComputeCoarseLocalPosition;
};


#endif
