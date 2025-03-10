#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/ClusterizerAlgo.dev.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  class clusterizeKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  portablevertex::TrackDeviceCollection::View tracks,
                                  portablevertex::VertexDeviceCollection::View vertices,
                                  const portablevertex::ClusterParamsHostCollection::ConstView cParams,
                                  double* beta_,
                                  double* osumtkwt_,
                                  int trackBlockSize) const {
      // Core of the clusterization algorithm
      // Produces set of clusters for input set of block-overlapped tracks
      // tracks contains input track parameters and includes the track-vertex assignment modified during this kernel
      // vertices is filled up by this kernel with protocluster properties
      // beta_ and osumtkwt_ are used to pass the final values of _beta and _osumtkwt on each block to the next kernel
      int blockSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
          acc)[0u];  // In GPU blockSize and trackBlockSize should be identical from how the kernel is called, in CPU not
      int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  // Thread number inside block
      int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];     // Block number inside grid
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgo::operator()] Start clustering block %i\n", blockIdx);
        printf("[ClusterizerAlgo::operator()] Parameters blockSize %i, trackBlockSize %i\n", blockSize, trackBlockSize);
      }
#endif
      double& _beta =
          alpaka::declareSharedVar<double, __COUNTER__>(acc);  // 1/T in the annealing loop, shared in the block
      double& _osumtkwt = alpaka::declareSharedVar<double, __COUNTER__>(
          acc);  // Sum of all track weights for normalization of probabilities, shared in the block
      for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
           itrack += blockSize) {
        double temp_weight = static_cast<double>(tracks[itrack].weight());
        alpaka::atomicAdd(acc, &_osumtkwt, temp_weight, alpaka::hierarchy::Threads{});
      }
      alpaka::syncBlockThreads(acc);
      if (once_per_block(acc)) {
        _osumtkwt = 1. / _osumtkwt;
      }
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgo::operator()] BlockIdx %i, _osumtkwt=%1.3f \n", blockIdx, _osumtkwt);
      }
#endif
      alpaka::syncBlockThreads(acc);
      // In each block, initialize to a single vertex with all tracks
      initialize(acc, tracks, vertices, cParams, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgo::operator()] BlockIdx %i, vertices initialized\n", blockIdx);
      }
#endif
      alpaka::syncBlockThreads(acc);
      // First estimation of critical temperature
      getBeta0(acc, tracks, vertices, cParams, _beta, trackBlockSize);
      alpaka::syncBlockThreads(acc);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgo::operator()] BlockIdx %i, first estimation of TC _beta=%1.3f \n", blockIdx, _beta);
      }
#endif
      // Cool down to betamax with rho = 0.0 (no regularization term)
      thermalize(acc, tracks, vertices, cParams, _osumtkwt, _beta, cParams.delta_highT(), 0.0, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgo::operator()] BlockIdx %i, first thermalization ended\n", blockIdx);
      }
#endif
      // Now the cooling loop
      coolingWhileSplitting(acc, tracks, vertices, cParams, _osumtkwt, _beta, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgo::operator()] BlockIdx %i, cooling ended, T at stop _beta=%1.3f\n", blockIdx, _beta);
      }
#endif
      // After cooling, merge closeby vertices
      reMergeTracks(acc, tracks, vertices, cParams, _osumtkwt, _beta, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgo::operator()] BlockIdx %i, merge, last merging step done\n", blockIdx);
      }
#endif
      if (once_per_block(acc)) {
        beta_[blockIdx] = _beta;
        osumtkwt_[blockIdx] = _osumtkwt;
      }
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgo::operator()] BlockIdx %i end\n", blockIdx);
      }
#endif
    }
  };  // clusterizeKernel

  ClusterizerAlgo::ClusterizerAlgo(Queue& queue, int32_t bSize)
      : beta_(cms::alpakatools::make_device_buffer<double[]>(queue, bSize)),
        osumtkwt_(cms::alpakatools::make_device_buffer<double[]>(queue, bSize)) {
    alpaka::memset(queue, beta_, bSize);
    alpaka::memset(queue, osumtkwt_, bSize);
  }

  void ClusterizerAlgo::clusterize(Queue& queue,
                                   portablevertex::TrackDeviceCollection& deviceTrack,
                                   portablevertex::VertexDeviceCollection& deviceVertex,
                                   const std::shared_ptr<portablevertex::ClusterParamsHostCollection> cParams,
                                   int32_t nBlocks,
                                   int32_t blockSize) {
    const int blocks = divide_up_by(nBlocks * blockSize, blockSize);  //nBlocks of size blockSize
    alpaka::exec<Acc1D>(
        queue,
        make_workdiv<Acc1D>(blocks, blockSize),
        clusterizeKernel{},
        deviceTrack
            .view(),  // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
        deviceVertex.view(),
        cParams->view(),
        beta_.data(),
        osumtkwt_.data(),
        blockSize);
  }  // ClusterizerAlgo::clusterize
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
