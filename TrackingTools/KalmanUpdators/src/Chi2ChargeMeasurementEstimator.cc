#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2ChargeMeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

bool Chi2ChargeMeasurementEstimator::checkClusterCharge(DetId id, const SiStripCluster  & cluster, const TrajectoryStateOnSurface& ts) const
{
    return siStripClusterTools::chargePerCM(id, cluster.amplitudes().begin(), cluster.amplitudes().end(), ts.localParameters() ) >  minGoodStripCharge_;
}

bool Chi2ChargeMeasurementEstimator::checkCharge(const TrackingRecHit& aRecHit, const TrajectoryStateOnSurface& ts) const
{
  auto const & hit = aRecHit;
  SiStripDetId id = aRecHit.geographicalId();

  if (aRecHit.getRTTI() == 4) {
    const SiStripMatchedRecHit2D & matchHit = static_cast<const SiStripMatchedRecHit2D &>(hit);
    return checkClusterCharge(id, matchHit.monoCluster(),ts ) && checkClusterCharge(id, matchHit.stereoCluster(),ts);
  } else {
    auto const & thit = static_cast<const BaseTrackerRecHit &>(hit);
    auto const & clus = thit.firstClusterRef();
    return checkClusterCharge(id, *clus.cluster_strip(), ts);
  }
}


bool Chi2ChargeMeasurementEstimator::preFilter(const TrajectoryStateOnSurface& ts,
                                               const TrackingRecHit& hit) const {
  auto detid = hit.geographicalId();

  if (ts.globalMomentum().perp2()>pTChargeCutThreshold2_) return true;
 
  uint32_t subdet = detid.subdetId();

  if ( (subdet>2)&&cutOnStripCharge_) return checkCharge(hit, ts);


  /*  pixel charge not implemented as not used...
     auto const & thit = static_cast<const SiPixelRecHit &>(hit);
     thit.cluster()->charge() ...

  */

  return true;
}
