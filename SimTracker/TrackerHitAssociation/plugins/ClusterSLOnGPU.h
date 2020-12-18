#ifndef SimTracker_TrackerHitAssociation_plugins_ClusterSLOnGPU_h
#define SimTracker_TrackerHitAssociation_plugins_ClusterSLOnGPU_h

#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h"

namespace clusterSLOnGPU {

  using ClusterSLView = trackerHitAssociationHeterogeneous::ClusterSLView;
  using Clus2TP = ClusterSLView::Clus2TP;
  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DCUDA;

  class Kernel {
  public:
    explicit Kernel(bool dump);
    ~Kernel() {}
    trackerHitAssociationHeterogeneous::ProductCUDA makeAsync(SiPixelDigisCUDA const& dd,
                                                              uint32_t ndigis,
                                                              HitsOnCPU const& hh,
                                                              Clus2TP const* digi2tp,
                                                              uint32_t nhits,
                                                              uint32_t nlinks,
                                                              cudaStream_t stream) const;

  private:
  public:
    bool doDump;
  };
}  // namespace clusterSLOnGPU

#endif  // SimTracker_TrackerHitAssociation_plugins_ClusterSLOnGPU_h
