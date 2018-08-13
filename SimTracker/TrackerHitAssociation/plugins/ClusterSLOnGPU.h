#ifndef SimTrackerTrackerHitAssociationClusterSLOnGPU_H
#define SimTrackerTrackerHitAssociationClusterSLOnGPU_H

#include <cuda_runtime.h>
#include <cuda/api_wrappers.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


#include "trackerHitAssociationHeterogeneousProduct.h"

#include "RecoLocalTracker/SiPixelClusterizer/plugins/siPixelRawToClusterHeterogeneousProduct.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"




namespace clusterSLOnGPU {

  using ClusterSLGPU = trackerHitAssociationHeterogeneousProduct::ClusterSLGPU;
  using GPUProduct = trackerHitAssociationHeterogeneousProduct::GPUProduct;

  using DigisOnGPU = siPixelRawToClusterHeterogeneousProduct::GPUProduct;
  using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
  using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;


  class Kernel {
  public:
    Kernel(cuda::stream_t<>& stream, bool dump);
    ~Kernel() {deAlloc();}
    void algo(DigisOnGPU const & dd, uint32_t ndigis, HitsOnCPU const & hh, uint32_t nhits, uint32_t n, cuda::stream_t<>& stream);
    GPUProduct getProduct() { return GPUProduct{slgpu.me_d};}
    
  private:
     void alloc(cuda::stream_t<>& stream);
     void deAlloc(); 
     void zero(cudaStream_t stream);
  public:
     ClusterSLGPU slgpu; 
     bool doDump;
  };
}

#endif
