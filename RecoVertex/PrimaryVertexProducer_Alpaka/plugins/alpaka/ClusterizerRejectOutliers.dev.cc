#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/ClusterizerAlgo.dev.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  class rejectOutliersKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  portablevertex::TrackDeviceCollection::View tracks,
                                  portablevertex::VertexDeviceCollection::View vertices,
                                  const portablevertex::ClusterParamsHostCollection::ConstView cParams,
                                  double* beta_,
                                  double* osumtkwt_,
                                  int trackBlockSize) const {
      // This has the core of the clusterization algorithm
      //int blockSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      //int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; // Thread number inside block
      int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];  // Block number inside grid

      double& _beta = alpaka::declareSharedVar<double, __COUNTER__>(acc);
      double& osumtkwt = alpaka::declareSharedVar<double, __COUNTER__>(acc);

      if (once_per_block(acc)) {
        _beta = beta_[blockIdx];
        osumtkwt = osumtkwt_[blockIdx];
      }
      alpaka::syncBlockThreads(acc);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgoRejectOutliers::operator()] Start outlier rejection for block %i\n", blockIdx);
        printf("[ClusterizerAlgoRejectOutliers::operator()] Parameter, trackBlockSize %i\n", trackBlockSize);
      }
#endif
      // After splitting we might get some candidates that are very low quality/have very far away tracks
      rejectOutliers(acc, tracks, vertices, cParams, osumtkwt, _beta, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgoRejectOutliers::operator()] End outlier rejection for block %i\n", blockIdx);
      }
#endif
      alpaka::syncBlockThreads(acc);
    }
  };  // class kernel

  void ClusterizerAlgo::reject_outliers(Queue& queue,
                                        portablevertex::TrackDeviceCollection& deviceTrack,
                                        portablevertex::VertexDeviceCollection& deviceVertex,
                                        const std::shared_ptr<portablevertex::ClusterParamsHostCollection> cParams,
                                        int32_t nBlocks,
                                        int32_t blockSize) {
    const int blocks = divide_up_by(nBlocks * blockSize, blockSize);  //nBlocks of size blockSize
    alpaka::exec<Acc1D>(
        queue,
        make_workdiv<Acc1D>(blocks, blockSize),
        rejectOutliersKernel{},
        deviceTrack
            .view(),  // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
        deviceVertex.view(),
        cParams->view(),
        beta_.data(),
        osumtkwt_.data(),
        blockSize);
  }  // ClusterizerAlgo::reject_outliers
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
