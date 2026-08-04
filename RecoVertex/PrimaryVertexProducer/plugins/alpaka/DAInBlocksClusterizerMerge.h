#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerMerge_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerMerge_h

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

  ALPAKA_FN_ACC static void merge(const Acc1D& acc,
                                  reco::TrackForVertexDeviceCollection::View tracks,
                                  reco::VertexDeviceCollection::View vertices,
                                  DAInBlocksClusterParameters const cParams,
                                  double& osumtkwt,
                                  double& beta,
                                  int trackBlockSize) {
    // If two vertex are too close together, merge them
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int nprev = vertices[blockIdx].nV();
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
    if (once_per_block(acc)) {
      printf("[DAInBlocksClusterizerAlgo::merge()] BlockIdx %i, start merging \n", blockIdx);
    }
#endif
    if (nprev < 2)
      return;
    for (int ivertexO : uniform_elements(acc, round_up_by(vertices[blockIdx].nV(), alpaka::warp::getSize(acc)))) {
      if (ivertexO < vertices[blockIdx].nV()){
        int ivertex = vertices[maxVerticesPerBlock * blockIdx + ivertexO].order();
        int ivertexnext = vertices[maxVerticesPerBlock * blockIdx + ivertexO + 1].order();
        vertices[ivertex].aux1() = abs(vertices[ivertex].z() - vertices[ivertexnext].z());
      }
    }
    alpaka::syncBlockThreads(acc);
    // Sorter things
    auto& critical_dist = alpaka::declareSharedVar<float[128], __COUNTER__>(acc);
    auto& critical_index = alpaka::declareSharedVar<float[128], __COUNTER__>(acc);
    int& ncritical = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    if (once_per_block(acc)) {
      ncritical = 0;
      for (int ivertexO = maxVerticesPerBlock * blockIdx;
           ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV() - 1;
           ivertexO += 1) {
        int ivertex = vertices[ivertexO].order();
        if (vertices[ivertex].aux1() < cParams.zmerge) {  // i.e., if we are to split the vertex
          critical_dist[ncritical] = abs(vertices[ivertex].aux1());
          critical_index[ncritical] = ivertexO;
          ncritical++;
          if (ncritical > 128)
            break;
        }
      }
    }  // end once_per_block
    alpaka::syncBlockThreads(acc);
    if (ncritical == 0)
      return;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
    if (once_per_block(acc)) {
      printf("[DAInBlocksClusterizerAlgo::merge()] BlockIdx %i, %i vertices to be merged\n", blockIdx, ncritical);
    }
#endif
    if (ncritical == 0 || maxVerticesPerBlock == nprev) {
      return;
    } else {
      // All threads are running the same code, to know when to exit
      if (ncritical == 0 || maxVerticesPerBlock == nprev)
        return;
      int ikO = 0;
      double minVal = 999999.;
      for (int sort1 = 0; sort1 < ncritical; ++sort1) {
        if (critical_dist[sort1] > minVal) {
          minVal = critical_dist[sort1];
          ikO = sort1;
        }
      }
      critical_dist[ikO] = minVal;
      int ivertexO = critical_index[ikO];
      int ivertex = vertices[ivertexO].order();
      int ivertexnext = blockIdx * maxVerticesPerBlock + nprev - 1;
      // A little bit of safety here. First is needed to avoid reading the -1 entry of vertices->order. Second is only needed if we go over 1024 vertices
      if (ivertexO < blockIdx * maxVerticesPerBlock + nprev - 1)
        ivertexnext = vertices[ivertexO + 1].order();
      alpaka::syncBlockThreads(acc);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::merge()] BlockIdx %i, merge vertex %i into vertex %i\n",
               blockIdx,
               ivertex,
               ivertexnext);
      }
#endif

      if (once_per_block(acc)) {
        vertices[ivertex].isGood() = false;  // Not deleting all the info, just disable it!
        double rho = vertices[ivertex].rho() + vertices[ivertexnext].rho();
        if (rho > 1.e-40) {
          vertices[ivertexnext].z() = (vertices[ivertex].rho() * vertices[ivertex].z() +
                                       vertices[ivertexnext].rho() * vertices[ivertexnext].z()) /
                                      rho;
        } else {
          vertices[ivertexnext].z() = 0.5 * (vertices[ivertex].z() + vertices[ivertexnext].z());
        }
        vertices[ivertexnext].rho() = rho;
        vertices[ivertexnext].sw() += vertices[ivertex].sw();
        for (int ivertexOO = ivertexO; ivertexOO < maxVerticesPerBlock * blockIdx + nprev; ++ivertexOO) {
          vertices[ivertexOO].order() = vertices[ivertexOO + 1].order();
        }
        vertices[blockIdx].nV() = vertices[blockIdx].nV() - 1;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
        if (once_per_block(acc)) {
          printf("[DAInBlocksClusterizerAlgo::merge()] BlockIdx %i, merged vertex %i with z=%1.3f,rho=%1.3f\n",
                 blockIdx,
                 ivertexnext,
                 vertices[ivertexnext].z(),
                 vertices[ivertexnext].rho());
        }
#endif
      }  // end once_per_block
      alpaka::syncBlockThreads(acc);
      for (int resort = 0; resort < ncritical; ++resort) {
        if (critical_index[resort] > ivertexO)
          critical_index[resort]--;  // critical_index refers to the original vertices->order, so it needs to be updated
      }
      nprev = vertices[blockIdx].nV();  // And to the counter of previous vertices
      for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize,alpaka::warp::getSize(acc)))) {
	if (itrackO < trackBlockSize){
          int itrack = itrackO + blockIdx * trackBlockSize;
          if (tracks[itrack].kmax() > ivertexO + 1)
            tracks[itrack].kmax()--;
          if ((tracks[itrack].kmin() > ivertexO) || ((tracks[itrack].kmax() < (tracks[itrack].kmin() + 1)) &&
                                                     (tracks[itrack].kmin() > maxVerticesPerBlock * blockIdx)))
            tracks[itrack].kmin()--;
	}
      }
    }
    alpaka::syncBlockThreads(acc);
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerMerge_h
