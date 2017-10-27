#ifndef RECOTRACKER_TRANSIENTRECHITBUILDER_H
#define RECOTRACKER_TRANSIENTRECHITBUILDER_H

#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"



class SiStripRecHitMatcher;
class PixelClusterParameterEstimator;
class StripClusterParameterEstimator;
class Phase2StripCPE;


class TkTransientTrackingRecHitBuilder final : public TransientTrackingRecHitBuilder {
  
 public:
  TkTransientTrackingRecHitBuilder (const TrackingGeometry* trackingGeometry, 
				    const PixelClusterParameterEstimator * ,
				    const StripClusterParameterEstimator * ,
                                    const SiStripRecHitMatcher           *,
				    bool computeCoarseLocalPositionFromDisk);
  TkTransientTrackingRecHitBuilder (const TrackingGeometry* trackingGeometry, 
				    const PixelClusterParameterEstimator * ,
				    const ClusterParameterEstimator<Phase2TrackerCluster1D> * );

  TransientTrackingRecHit::RecHitPointer build (const TrackingRecHit * p) const override ;


  const PixelClusterParameterEstimator * pixelClusterParameterEstimator() const {return pixelCPE;}
  const StripClusterParameterEstimator * stripClusterParameterEstimator() const {return stripCPE;}
  const ClusterParameterEstimator<Phase2TrackerCluster1D> * phase2TrackerClusterParameterEstimator() const {return phase2OTCPE;}
  const SiStripRecHitMatcher           * siStripRecHitMatcher() const {return theMatcher;}
  const TrackingGeometry               * geometry() const  { return tGeometry_;}

  // for the time being here...
  TkClonerImpl cloner() const { 
    if(phase2OTCPE == nullptr)
      return TkClonerImpl(pixelCPE,stripCPE,theMatcher);
    else
      return TkClonerImpl(pixelCPE,phase2OTCPE);
  }

private:


 private:
  const TrackingGeometry* tGeometry_;
  const PixelClusterParameterEstimator * pixelCPE;
  const StripClusterParameterEstimator * stripCPE;
  const SiStripRecHitMatcher           * theMatcher;
  bool theComputeCoarseLocalPosition;
  const ClusterParameterEstimator<Phase2TrackerCluster1D> * phase2OTCPE;
};


#endif
