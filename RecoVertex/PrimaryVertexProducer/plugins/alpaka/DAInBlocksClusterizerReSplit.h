#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerReSplit_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerReSplit_h

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

  ALPAKA_FN_ACC static void reSplitTracks(const Acc1D& acc,
                                          reco::TrackForVertexDeviceCollection::View tracks,
                                          reco::VertexDeviceCollection::View vertices,
                                          DAInBlocksClusterParameters const cParams,
                                          double& osumtkwt,
                                          double& beta,
                                          int trackBlockSize) {
    // Last splitting at the minimal temperature which is a bit more permissive
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int ntry = 0;
    double threshold = 1.0;
    int nprev = vertices[blockIdx].nV();
    split(acc, tracks, vertices, cParams, osumtkwt, beta, threshold, trackBlockSize);
    while (nprev != vertices[blockIdx].nV() && (ntry++ < 10)) {
      thermalize(acc, tracks, vertices, cParams, osumtkwt, beta, cParams.delta_highT, 0.0, trackBlockSize);
      nprev = vertices[blockIdx].nV();
      merge(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
      while (nprev != vertices[blockIdx].nV()) {
        nprev = vertices[blockIdx].nV();
        update(acc, tracks, vertices, cParams, osumtkwt, beta, 0.0, false, trackBlockSize);
        merge(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
      }
      threshold *= 1.1;
      split(acc, tracks, vertices, cParams, osumtkwt, beta, threshold, trackBlockSize);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerReSplit_h
