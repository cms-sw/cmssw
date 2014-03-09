#include "RecoTracker/TransientTrackingRecHit/interface/TKClonerImpl.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"



SiPixelRecHit * TKClonerImpl::operator()(SiPixelRecHit const & hit, TrajectoryStateOnSurface& tsos) const {
  const SiPixelCluster& clust = *hit.cluster();  
  PixelClusterParameterEstimator::LocalValues lv = 
    theCPE->localParameters( clust, hit.detUnit(), tsos);
  return new SiPixelRecHit(lv.first, lv.second, theCPE->rawQualityWord(), hit.det()->geographicalId() hit.det(), clus());

}

SiStripRecHit2D * TKClonerImpl::operator()(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const {


}



SiStripRecHit1D * TKClonerImpl::operator()(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const {


}



SiStripMatchedRecHit2D * TKClonerImpl::operator()(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const {


}
