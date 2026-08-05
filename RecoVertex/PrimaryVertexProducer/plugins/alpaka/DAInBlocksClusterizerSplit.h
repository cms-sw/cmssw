#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerSplit_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerSplit_h

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

  ALPAKA_FN_ACC static void split(const Acc1D& acc,
                                  reco::TrackForVertexDeviceCollection::View tracks,
                                  reco::VertexDeviceCollection::View vertices,
                                  DAInBlocksClusterParameters const cParams,
                                  double& osumtkwt,
                                  double& beta,
                                  double threshold,
                                  int trackBlockSize) {
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    update(acc, tracks, vertices, cParams, osumtkwt, beta, 0.0, true, trackBlockSize);
    double epsilon = 1e-3;
    int nprev = vertices[blockIdx].nV();
    // Set critical T for all vertices
    for (int ivertexO : uniform_elements(acc, round_up_by(vertices[blockIdx].nV(), alpaka::warp::getSize(acc)))) {
      if (ivertexO < vertices[blockIdx].nV()) {
        int ivertex = vertices[maxVerticesPerBlock * blockIdx + ivertexO].order();
        double Tc = 2 * vertices[ivertex].swE() / vertices[ivertex].sw();
        vertices[ivertex].aux1() = Tc;
      }
    }
    alpaka::syncBlockThreads(acc);
    auto& critical_temp = alpaka::declareSharedVar<float[128], __COUNTER__>(acc);
    auto& critical_index = alpaka::declareSharedVar<float[128], __COUNTER__>(acc);
    int& ncritical = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    // Information for the vertex splitting properties
    double& p1 = alpaka::declareSharedVar<double, __COUNTER__>(acc);
    double& p2 = alpaka::declareSharedVar<double, __COUNTER__>(acc);
    double& z1 = alpaka::declareSharedVar<double, __COUNTER__>(acc);
    double& z2 = alpaka::declareSharedVar<double, __COUNTER__>(acc);
    double& w1 = alpaka::declareSharedVar<double, __COUNTER__>(acc);
    double& w2 = alpaka::declareSharedVar<double, __COUNTER__>(acc);

    if (once_per_block(acc)) {
      ncritical = 0;
      for (int ivertexO = maxVerticesPerBlock * blockIdx;
           ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV();
           ++ivertexO) {
        int ivertex = vertices[ivertexO].order();
        if (vertices[ivertex].aux1() * beta > threshold) {
          critical_temp[ncritical] = abs(vertices[ivertex].aux1());
          critical_index[ncritical] = ivertexO;
          ncritical++;
          if (ncritical > 128)
            break;
        }
      }
    }  // end once_per_block
    alpaka::syncBlockThreads(acc);
    if (ncritical == 0 || maxVerticesPerBlock == nprev)
      return;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
    if (once_per_block(acc)) {
      printf("[DAInBlocksClusterizerAlgo::split()] BlockIdx %i, split %i vertices\n", blockIdx, ncritical);
    }
#endif
    for (int sortO = 0; sortO < ncritical; ++sortO) {  // All threads are running the same code, to know when to exit
      if (ncritical == 0 || maxVerticesPerBlock == nprev)
        return;
      int ikO = 0;
      double maxVal = -1.;
      for (int sort1 = 0; sort1 < ncritical; ++sort1) {
        if (critical_temp[sort1] > maxVal) {
          maxVal = critical_temp[sort1];
          ikO = sort1;
        }
      }
      critical_temp[ikO] = -1.;
      int ivertexO = critical_index[ikO];
      int ivertex = vertices[ivertexO].order();  // This will be splitted
      int ivertexprev = blockIdx * maxVerticesPerBlock;
      int ivertexnext = blockIdx * maxVerticesPerBlock + nprev - 1;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::split()] BlockIdx %i, splitting vertex %i\n", blockIdx, ivertex);
      }
#endif
      // Safety here. First is needed to avoid reading the -1 entry of vertices->order. Second in case we go over 511 vertices, but better keep it just in case
      if (ivertexO > blockIdx * maxVerticesPerBlock)
        ivertexprev = vertices[ivertexO - 1].order();
      if (ivertexO < blockIdx * maxVerticesPerBlock + nprev - 1)
        ivertexnext = vertices[ivertexO + 1].order();
      if (once_per_block(acc)) {
        p1 = 0.;
        p2 = 0.;
        z1 = 0.;
        z2 = 0.;
        w1 = 0.;
        w2 = 0.;
      }
      alpaka::syncBlockThreads(acc);
      for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize, alpaka::warp::getSize(acc)))) {
        if (itrackO < trackBlockSize) {
          int itrack = itrackO + blockIdx * trackBlockSize;
          if (tracks[itrack].sum_Z() > 1e-40) {
            // winner-takes-all, usually overestimates splitting
            double tl = tracks[itrack].z() < vertices[ivertex].z() ? 1. : 0.;
            double tr = 1. - tl;
            // soften it, especially at low T
            double arg = (tracks[itrack].z() - vertices[ivertex].z()) * sqrt((beta)*tracks[itrack].oneoverdz2());
            if (abs(arg) < 20) {
              double t = exp(-arg);
              tl = t / (t + 1.);
              tr = 1 / (t + 1.);
            }
            // Recompute split vertex quantities
            double p = vertices[ivertex].rho() * tracks[itrack].weight() *
                       exp(-(beta) * (tracks[itrack].z() - vertices[ivertex].z()) *
                           (tracks[itrack].z() - vertices[ivertex].z()) * tracks[itrack].oneoverdz2()) /
                       tracks[itrack].sum_Z();
            double w = p * tracks[itrack].oneoverdz2();
            alpaka::atomicAdd(acc, &p1, p * tl, alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(acc, &p2, p * tr, alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(acc, &z1, w * tl * tracks[itrack].z(), alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(acc, &z2, w * tr * tracks[itrack].z(), alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(acc, &w1, w * tl, alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(acc, &w2, w * tr, alpaka::hierarchy::Threads{});
          }
        }
      }
      alpaka::syncBlockThreads(acc);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf(
            "[DAInBlocksClusterizerAlgo::split()] BlockIdx %i, for vertex %i, p1=%1.3f, p2=%1.3f, w1=%1.3f, w2=%1.3f, "
            "z1=%1.3f, "
            "z2=%1.3f\n",
            blockIdx,
            ivertex,
            p1,
            p2,
            w1,
            w2,
            z1,
            z2);
      }
#endif

      if (once_per_block(acc)) {
        // If one vertex is taking all the things, then set the others slightly off to help splitting
        if (w1 > 0) {
          z1 = z1 / w1;
        } else {
          z1 = vertices[ivertex].z() - epsilon;
        }
        if (w2 > 0) {
          z2 = z2 / w2;
        } else {
          z2 = vertices[ivertex].z() + epsilon;
        }
        // If there is not enough room, reduce split size
        if ((ivertexO > maxVerticesPerBlock * blockIdx) &&
            (z1 <
             (0.6 * vertices[ivertex].z() +
              0.4 *
                  vertices[ivertexprev]
                      .z()))) {  // First in the if is ivertexO, as we care on whether the vertex is the leftmost or rightmost
          z1 = 0.6 * vertices[ivertex].z() + 0.4 * vertices[ivertexprev].z();
        }
        if ((ivertexO < maxVerticesPerBlock * blockIdx + nprev - 1) &&
            (z2 > (0.6 * vertices[ivertex].z() + 0.4 * vertices[ivertexnext].z()))) {
          z2 = 0.6 * vertices[ivertex].z() + 0.4 * vertices[ivertexnext].z();
        }
      }  // end once_per_block
      alpaka::syncBlockThreads(acc);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::split()] BlockIdx %i, vertex %i will split into z1=%1.3f, z2=%1.3f\n",
               blockIdx,
               ivertex,
               z1,
               z2);
      }
#endif
      const int breaknnew = 999999;
      int nnew = breaknnew;
      // Find the first empty index to save the vertex
      for (int icheck = maxVerticesPerBlock * blockIdx; icheck < maxVerticesPerBlock * (blockIdx + 1); icheck++) {
        if (not(vertices[icheck].isGood())) {
          nnew = icheck;
          break;
        }
      }
      if (nnew == breaknnew)
        break;  // Need to check if we exhausted the list of vertices to split in all threads so we exit in all of them properly
      if (once_per_block(acc)) {
        double pk1 = p1 * vertices[ivertex].rho() / (p1 + p2);
        double pk2 = p2 * vertices[ivertex].rho() / (p1 + p2);
        vertices[ivertex].z() = z2;
        vertices[ivertex].rho() = pk2;
        // Insert it into the first available slot
        vertices[nnew].z() = z1;
        vertices[nnew].rho() = pk1;
        vertices[nnew].isGood() = true;
        // This is likely not needed as far as it is reset anytime we call update but better be safe in case we reenable a previously disable vertex (i.e. split in the memory where a merged one was)
        vertices[nnew].sw() = 0.;
        vertices[nnew].swE() = 0.;
        for (int ivnew = maxVerticesPerBlock * blockIdx + nprev; ivnew > ivertexO;
             ivnew--) {  // As we add a vertex, we update from the back downwards
          vertices[ivnew].order() = vertices[ivnew - 1].order();
        }
        vertices[ivertexO].order() = nnew;
        vertices[blockIdx].nV() += 1;
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
        if (once_per_block(acc)) {
          printf("[DAInBlocksClusterizerAlgo::split()] BlockIdx %i, vertex %i did split into indexes %i and %i\n",
                 blockIdx,
                 ivertex,
                 ivertex,
                 nnew);
          dump(acc, beta, vertices);
        }
#endif
      }
      alpaka::syncBlockThreads(acc);
      // Now, update kmin/kmax for all tracks
      for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize, alpaka::warp::getSize(acc)))) {
        if (itrackO < trackBlockSize) {
          int itrack = itrackO + blockIdx * trackBlockSize;
          if (tracks[itrack].kmin() > ivertexO)
            tracks[itrack].kmin()++;
          if ((tracks[itrack].kmax() >= ivertexO) || (tracks[itrack].kmax() == tracks[itrack].kmin()))
            tracks[itrack].kmax()++;
        }
      }
      nprev = vertices[blockIdx].nV();
      if (once_per_block(acc)) {
        // If we did a splitting or old sorted list of vertex index is scrambled, so we need to fix it
        for (int resort = 0; resort < ncritical; ++resort) {
          if (critical_index[resort] > ivertexO)
            critical_index[resort]++;
        }
      }
      alpaka::syncBlockThreads(acc);
    }
    alpaka::syncBlockThreads(acc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerSplit_h
