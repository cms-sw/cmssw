#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerCooling_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerCooling_h

#include "DAInBlocksClusterizerAlgo.h"
#include "DAInBlocksClusterizerMerge.h"
#include "DAInBlocksClusterizerSplit.h"
#include "DAInBlocksClusterizerThermalize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"

//#ifndef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
//#define DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO 5
//#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  ALPAKA_FN_ACC static void coolingWhileSplitting(const Acc1D& acc,
                                                  reco::TrackForVertexDeviceCollection::View tracks,
                                                  reco::VertexDeviceCollection::View vertices,
                                                  DAInBlocksClusterParameters const cParams,
                                                  double& osumtkwt,
                                                  double& beta,
                                                  int trackBlockSize) {
    // Perform cooling of the deterministic annealing
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    double betafreeze = (1. / cParams.Tmin) * alpaka::math::sqrt(acc, cParams.coolingFactor);
    while (beta < betafreeze) {
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf(
            "[DAInBlocksClusterizerAlgo::coolingWhileSplitting()] BlockIdx %i, current beta=%1.8f\n", blockIdx, beta);
      }
#endif
      int nprev = vertices[blockIdx].nV();
      merge(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
      while (nprev != vertices[blockIdx].nV()) {
        // If we are here, we merged before, keep merging until stable
        nprev = vertices[blockIdx].nV();
        update(acc, tracks, vertices, cParams, osumtkwt, beta, 0.0, false, trackBlockSize);
        merge(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
      }  // end while after merging
      split(acc, tracks, vertices, cParams, osumtkwt, beta, 1.0, trackBlockSize);
      if (once_per_block(acc)) {
        // Cool down
        beta = beta / cParams.coolingFactor;
      }
      alpaka::syncBlockThreads(acc);
      thermalize(acc, tracks, vertices, cParams, osumtkwt, beta, cParams.delta_highT, 0.0, trackBlockSize);
    }
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    update(acc, tracks, vertices, cParams, osumtkwt, beta, 0.0, false, trackBlockSize);
  }  // end coolingWhileSplitting

  ALPAKA_FN_ACC static void reMergeTracks(const Acc1D& acc,
                                          reco::TrackForVertexDeviceCollection::View tracks,
                                          reco::VertexDeviceCollection::View vertices,
                                          DAInBlocksClusterParameters const cParams,
                                          double& osumtkwt,
                                          double& beta,
                                          int trackBlockSize) {
    // After the cooling, we merge any closeby vertices
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int nprev = vertices[blockIdx].nV();
    merge(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    while (nprev != vertices[blockIdx].nV()) {
      // Keep merging until stable
      set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
      update(acc, tracks, vertices, cParams, osumtkwt, beta, 0.0, false, trackBlockSize);
      nprev = vertices[blockIdx].nV();
      merge(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    }  // end while
  }  // end reMergeTracks

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerCooling_h
