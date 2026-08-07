#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerReject_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerReject_h

#include "DAInBlocksClusterizerAlgo.h"
#include "DAInBlocksClusterizerMerge.h"
#include "DAInBlocksClusterizerPurge.h"
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

  ALPAKA_FN_ACC static void rejectOutliers(const Acc1D& acc,
                                           reco::TrackForVertexDeviceCollection::View tracks,
                                           reco::VertexDeviceCollection::View vertices,
                                           DAInBlocksClusterParameters const cParams,
                                           double& osumtkwt,
                                           double& beta,
                                           int trackBlockSize) {
    // Treat outliers, either low quality vertex, or those with very far away tracks
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    double rho0 = 0.0;
    if (cParams.dzCutOff > 0) {
      rho0 = vertices[blockIdx].nV() > 1 ? 1. / vertices[blockIdx].nV() : 1.;
      for (int rhoindex = 0; rhoindex < 5; rhoindex++) {
        update(acc, tracks, vertices, cParams, osumtkwt, beta, rhoindex * rho0 / 5., false, trackBlockSize);
      }
    }  // end if
    thermalize(acc, tracks, vertices, cParams, osumtkwt, beta, cParams.delta_lowT, rho0, trackBlockSize);
    int nprev = vertices[blockIdx].nV();
    merge(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    while (nprev != vertices[blockIdx].nV()) {
      set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
      update(acc, tracks, vertices, cParams, osumtkwt, beta, rho0, false, trackBlockSize);
      nprev = vertices[blockIdx].nV();
      merge(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    }
    while (beta < 1. / cParams.Tpurge) {
      alpaka::syncBlockThreads(acc);
      if (once_per_block(acc)) {
        beta = alpaka::math::min(acc, beta / cParams.coolingFactor, 1. / cParams.Tpurge);
      }
      alpaka::syncBlockThreads(acc);
      thermalize(acc, tracks, vertices, cParams, osumtkwt, beta, cParams.delta_lowT, rho0, trackBlockSize);
    }
    alpaka::syncBlockThreads(acc);
    nprev = vertices[blockIdx].nV();
    purge(acc, tracks, vertices, cParams, osumtkwt, beta, rho0, trackBlockSize);
    while (nprev != vertices[blockIdx].nV()) {
      thermalize(acc, tracks, vertices, cParams, osumtkwt, beta, cParams.delta_lowT, rho0, trackBlockSize);
      nprev = vertices[blockIdx].nV();
      purge(acc, tracks, vertices, cParams, osumtkwt, beta, rho0, trackBlockSize);
      alpaka::syncBlockThreads(acc);
    }
    while (beta < 1. / cParams.Tstop) {
      alpaka::syncBlockThreads(acc);
      if (once_per_block(acc)) {
        beta = alpaka::math::min(acc, beta / cParams.coolingFactor, 1. / cParams.Tstop);
      }
      alpaka::syncBlockThreads(acc);
      thermalize(acc, tracks, vertices, cParams, osumtkwt, beta, cParams.delta_lowT, rho0, trackBlockSize);
    }
    alpaka::syncBlockThreads(acc);
    // The last track to vertex assignment!
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
  }  // rejectOutliers

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerReject_h
