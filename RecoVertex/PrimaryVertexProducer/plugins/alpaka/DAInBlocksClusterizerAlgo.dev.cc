#include "RecoVertex/PrimaryVertexProducer/plugins/alpaka/DAInBlocksClusterizerAlgo.dev.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  class ClusterizeKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TrackDeviceCollection::View tracks,
                                  VertexDeviceCollection::View vertices,
                                  ClusterParameters const& cParams,
                                  double* beta_,
                                  double* osumtkwt_,
                                  int trackBlockSize) const {
      // Core of the clusterization algorithm
      // Produces set of clusters for input set of block-overlapped tracks
      // tracks contains input track parameters and includes the track-vertex assignment modified during this kernel
      // vertices is filled up by this kernel with protocluster properties
      // beta_ and osumtkwt_ are used to pass the final values of beta and osumtkwt on each block to the next kernel
      int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(
          acc)[0u];  
      int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  
      int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];     
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::operator()] Start clustering block %i\n", blockIdx);
        printf("[DAInBlocksClusterizerAlgo::operator()] Parameters blockSize %i, trackBlockSize %i\n", blockSize, trackBlockSize);
      }
#endif
      double& beta =
          alpaka::declareSharedVar<double, __COUNTER__>(acc);  // 1/T in the annealing loop, shared in the block
      double& osumtkwt = alpaka::declareSharedVar<double, __COUNTER__>(
          acc);  // Sum of all track weights for normalization of probabilities, shared in the block
      alpaka::syncBlockThreads(acc);
      if (once_per_block(acc)) {
	  osumtkwt = 0;
      }
      alpaka::syncBlockThreads(acc);
      for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
           itrack += blockSize) {
	if (not (tracks[itrack].isGood())) continue;
        double temp_weight = static_cast<double>(tracks[itrack].weight());
        alpaka::atomicAdd(acc, &osumtkwt, temp_weight, alpaka::hierarchy::Threads{});
      }
      alpaka::syncBlockThreads(acc);
      if (once_per_block(acc)) {
        osumtkwt = 1. / osumtkwt;
      }
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::operator()] BlockIdx %i, osumtkwt=%1.3f \n", blockIdx, osumtkwt);
      }
#endif
      alpaka::syncBlockThreads(acc);
      // In each block, initialize to a single vertex with all tracks
      initialize(acc, tracks, vertices, cParams, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::operator()] BlockIdx %i, vertices initialized\n", blockIdx);
      }
#endif
      alpaka::syncBlockThreads(acc);
      // First estimation of critical temperature
      getBeta0(acc, tracks, vertices, cParams, beta, trackBlockSize);
      alpaka::syncBlockThreads(acc);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::operator()] BlockIdx %i, first estimation of TC beta=%1.8f \n", blockIdx, beta);
      }
#endif
      // Cool down to betamax with rho = 0.0 (no regularization term)
      thermalize(acc, tracks, vertices, cParams, osumtkwt, beta, cParams.delta_highT, 0.0, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::operator()] BlockIdx %i, first thermalization ended\n", blockIdx);
      }
#endif
      // Now the cooling loop
      coolingWhileSplitting(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::operator()] BlockIdx %i, cooling ended, T at stop beta=%1.8f\n", blockIdx, beta);
      }
#endif
      // After cooling, merge closeby vertices
      reMergeTracks(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::operator()] BlockIdx %i, merge, last merging step done\n", blockIdx);
      }
#endif
      if (once_per_block(acc)) {
        beta_[blockIdx] = beta;
        osumtkwt_[blockIdx] = osumtkwt;
      }
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::operator()] BlockIdx %i end\n", blockIdx);
      }
#endif
    }
  };  // ClusterizeKernel

  DAInBlocksClusterizerAlgo::DAInBlocksClusterizerAlgo(Queue& queue, int32_t bSize)
      : beta_(cms::alpakatools::make_device_buffer<double[]>(queue, bSize)),
        osumtkwt_(cms::alpakatools::make_device_buffer<double[]>(queue, bSize)) {
    alpaka::memset(queue, beta_, 0.);
    alpaka::memset(queue, osumtkwt_, 0.);
  }

  void DAInBlocksClusterizerAlgo::clusterize(Queue& queue,
                                   TrackDeviceCollection& deviceTrack,
                                   VertexDeviceCollection& deviceVertex,
                                   ClusterParameters const& cParams,
                                   int32_t nBlocks,
                                   int32_t blockSize) {
    alpaka::exec<Acc1D>(
        queue,
        make_workdiv<Acc1D>(nBlocks, 64),
        ClusterizeKernel{},
        deviceTrack
            .view(), 
        deviceVertex.view(),
        cParams,
        beta_.data(),
        osumtkwt_.data(),
        blockSize);
  }  // DAInBlocksClusterizerAlgo::clusterize
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
