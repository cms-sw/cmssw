#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"

#include "DAInBlocksClusterizerAlgo.h"
#include "DAInBlocksClusterizerArbitrate.h"

//#define DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR 1

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  class ArbitrateKernel {
  public:
    ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                  reco::TrackForVertexDeviceCollection::View tracks,
                                  reco::VertexDeviceCollection::View vertices,
                                  DAInBlocksClusterParameters const cParams,
                                  int32_t nBlocks) const {
      // This has the core of the clusterization algorithm
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgoArbitrator::operator()] Start arbitration\n");
      }
#endif
      resortVerticesAndAssign(acc, tracks, vertices, cParams);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgoArbitrator::operator()] Vertex reassignment finished\n");
      }
#endif
      alpaka::syncBlockThreads(acc);
      finalizeVertices(acc, tracks, vertices, cParams);  // In CUDA it used to be verticesAndClusterize
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgoArbitrator::operator()] Vertices finalized\n");
      }
#endif
      alpaka::syncBlockThreads(acc);
    }
  };  // class kernel

  void DAInBlocksClusterizerAlgo::arbitrate(Queue& queue,
                                            reco::TrackForVertexDeviceCollection& deviceTrack,
                                            reco::VertexDeviceCollection& deviceVertex,
                                            DAInBlocksClusterParameters const cParams,
                                            int32_t nBlocks,
                                            int32_t blockSize) {
    const int blocks = divide_up_by(blockSize, blockSize);
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, blockSize),
                        ArbitrateKernel{},
                        deviceTrack.view(),
                        deviceVertex.view(),
                        cParams,
                        nBlocks);
  }  // arbitraterAlgo::arbitrate

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
