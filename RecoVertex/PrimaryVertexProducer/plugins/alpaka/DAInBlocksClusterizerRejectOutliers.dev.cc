#include "RecoVertex/PrimaryVertexProducer/plugins/alpaka/DAInBlocksClusterizerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  class RejectOutliersKernel {
  public:
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TrackForVertexDeviceCollection::View tracks,
                                  VertexDeviceCollection::View vertices,
                                  ClusterParameters const& cParams,
                                  double* beta_,
                                  double* osumtkwt_,
                                  int trackBlockSize) const {
      int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];  // Block number inside grid

      double& beta = alpaka::declareSharedVar<double, __COUNTER__>(acc);
      double& osumtkwt = alpaka::declareSharedVar<double, __COUNTER__>(acc);

      if (once_per_block(acc)) {
        beta = beta_[blockIdx];
        osumtkwt = osumtkwt_[blockIdx];
      }
      alpaka::syncBlockThreads(acc);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgoRejectOutliers::operator()] Start outlier rejection for block %i\n",
               blockIdx);
        printf("[DAInBlocksClusterizerAlgoRejectOutliers::operator()] Parameter, trackBlockSize %i\n", trackBlockSize);
      }
#endif
      // After splitting we might get some candidates that are very low quality/have very far away tracks
      rejectOutliers(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgoRejectOutliers::operator()] End outlier rejection for block %i\n", blockIdx);
      }
#endif
      alpaka::syncBlockThreads(acc);
    }
  };  // class kernel

  void DAInBlocksClusterizerAlgo::reject_outliers(Queue& queue,
                                                  TrackForVertexDeviceCollection& deviceTrack,
                                                  VertexDeviceCollection& deviceVertex,
                                                  ClusterParameters const& cParams,
                                                  int32_t nBlocks,
                                                  int32_t blockSize) {
    const int blocks = divide_up_by(nBlocks * blockSize, blockSize);  //nBlocks of size blockSize
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, blockSize),
                        RejectOutliersKernel{},
                        deviceTrack.view(),
                        deviceVertex.view(),
                        cParams,
                        beta_.data(),
                        osumtkwt_.data(),
                        blockSize);
  }  // DAInBlocksClusterizerAlgo::reject_outliers
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
