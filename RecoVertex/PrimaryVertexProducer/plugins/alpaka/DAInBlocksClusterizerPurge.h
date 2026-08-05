#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerPurge_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerPurge_h

#include "DAInBlocksClusterizerAlgo.h"
#include "DAInBlocksClusterizerPrimitives.h"
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

  ALPAKA_FN_ACC static void purge(const Acc1D& acc,
                                  reco::TrackForVertexDeviceCollection::View tracks,
                                  reco::VertexDeviceCollection::View vertices,
                                  DAInBlocksClusterParameters const cParams,
                                  double& osumtkwt,
                                  double& beta,
                                  double rho0,
                                  int trackBlockSize) {
    // Remove repetitive or low quality entries
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    if (vertices[blockIdx].nV() < 2)
      return;
    double eps = 1e-40;
    int nunique_min = 2;
    double rhoconst = rho0 * exp(-beta * (cParams.dzCutOff * cParams.dzCutOff));
    int nprev = vertices[blockIdx].nV();
    // Reassign
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    for (int ivertexO : uniform_elements(acc, round_up_by(vertices[blockIdx].nV(), alpaka::warp::getSize(acc)))) {
      if (ivertexO < vertices[blockIdx].nV()) {
        int ivertex = vertices[maxVerticesPerBlock * blockIdx + ivertexO].order();
        vertices[ivertex].aux1() = 0;  // sum of track-vertex probabilities
        vertices[ivertex].aux2() = 0;  // number of uniquely assigned tracks
      }
    }
    alpaka::syncBlockThreads(acc);
    // Get quality of vertex in terms of #Tracks and sum of track probabilities
    for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize, alpaka::warp::getSize(acc)))) {
      if (itrackO < trackBlockSize) {
        int itrack = itrackO + blockIdx * trackBlockSize;
        double track_aux1 = ((tracks[itrack].sum_Z() > eps) && (tracks[itrack].weight() > cParams.uniquetrkminp))
                                ? 1. / tracks[itrack].sum_Z()
                                : 0.;
        for (int ivertexO = tracks[itrack].kmin(); ivertexO < tracks[itrack].kmax(); ++ivertexO) {
          int ivertex = vertices[ivertexO].order();
          double ppcut = cParams.uniquetrkweight * vertices[ivertex].rho() / (vertices[ivertex].rho() + rhoconst);
          double track_vertex_aux1 =
              exp(-(beta)*tracks[itrack].oneoverdz2() *
                  ((tracks[itrack].z() - vertices[ivertex].z()) * (tracks[itrack].z() - vertices[ivertex].z())));
          float p =
              vertices[ivertex].rho() * track_vertex_aux1 * track_aux1;  // The whole track-vertex P_ij = rho_j*p_ij*p_i
          alpaka::atomicAdd(acc, &vertices[ivertex].aux1(), p, alpaka::hierarchy::Threads{});
          if (p > ppcut) {
            alpaka::atomicAdd(acc, &vertices[ivertex].aux2(), 1.f, alpaka::hierarchy::Threads{});
          }
        }
      }
    }
    alpaka::syncBlockThreads(acc);
    // Find worst vertex to purge
    int& k0 = alpaka::declareSharedVar<int, __COUNTER__>(acc);

    if (once_per_block(acc)) {
      double sumpmin = tracks.nT();
      k0 = maxVerticesPerBlock * blockIdx + nprev;
      for (int ivertexO : uniform_elements(acc, round_up_by(vertices[blockIdx].nV(), alpaka::warp::getSize(acc)))) {
        if (ivertexO < vertices[blockIdx].nV()) {
          int ivertex = vertices[maxVerticesPerBlock * blockIdx + ivertexO].order();
          if ((vertices[ivertex].aux2() < nunique_min) && (vertices[ivertex].aux1() < sumpmin)) {
            sumpmin = vertices[ivertex].aux1();
            k0 = maxVerticesPerBlock * blockIdx + ivertexO;
          }
        }
      }  // end vertex for
      if (k0 != (int)(maxVerticesPerBlock * blockIdx + nprev)) {
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
        if (once_per_block(acc)) {
          printf("[DAInBlocksClusterizerAlgo::purge()] BlockIdx %i, some vertices need purging. Will start purging \n",
                 blockIdx);
        }
#endif
        for (int ivertexOO = k0; ivertexOO < maxVerticesPerBlock * blockIdx + (int)nprev - 1; ++ivertexOO) {
          vertices[ivertexOO].order() =
              vertices[ivertexOO + 1].order();  // Update vertex order taking out the purged one
        }
        vertices[blockIdx].nV()--;
        vertices[k0].isGood() = false;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
        if (once_per_block(acc)) {
          printf("[DAInBlocksClusterizerAlgo::purge()] BlockIdx %i, vertex %i purged\n", blockIdx, k0);
        }
#endif
      }
    }  // end once_per_block
    alpaka::syncBlockThreads(acc);
    if (k0 != (int)(maxVerticesPerBlock * blockIdx + (int)nprev)) {
      for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize, alpaka::warp::getSize(acc)))) {
        if (itrackO < trackBlockSize) {
          int itrack = itrackO + blockIdx * trackBlockSize;
          if (tracks[itrack].kmax() > k0)
            tracks[itrack].kmax()--;
          if ((tracks[itrack].kmin() > k0) || ((tracks[itrack].kmax() < (tracks[itrack].kmin() + 1)) &&
                                               (tracks[itrack].kmin() > (int)(maxVerticesPerBlock * blockIdx))))
            tracks[itrack].kmin()--;
        }
      }
    }  // end if
    alpaka::syncBlockThreads(acc);
    if (nprev != vertices[blockIdx].nV()) {
      set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerPurge_h
