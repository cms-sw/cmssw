#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerInitializers_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerInitializers_h

#include "DAInBlocksClusterizerAlgo.h"

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

  ALPAKA_FN_ACC static void initialize(const Acc1D& acc,
                                       reco::TrackForVertexDeviceCollection::View tracks,
                                       reco::VertexDeviceCollection::View vertices,
                                       DAInBlocksClusterParameters const cParams,
                                       int trackBlockSize) {
    // Initialize all vertices as empty, a single vertex in each block will be initialized with all tracks associated to it
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    vertices[blockIdx].nV() = 1;
    for (int ivertexO : uniform_elements(acc, round_up_by(maxVerticesPerBlock,alpaka::warp::getSize(acc)))) {
      if (ivertexO < maxVerticesPerBlock){
        int ivertex = ivertexO + maxVerticesPerBlock * blockIdx;
        vertices[ivertex].sw() = 0.;
        vertices[ivertex].swE() = 0.;
        vertices[ivertex].z() = 0.;
        vertices[ivertex].rho() = 0.;
        vertices[ivertex].isGood() = false;
        vertices[ivertex].order() = 9999;
        if (ivertex ==
            maxVerticesPerBlock *
                blockIdx) {  // Set up the initial single vetex containing everything which should only happen for the first vertex
          vertices[ivertex].rho() = 1.;
          vertices[ivertex].order() = maxVerticesPerBlock * blockIdx;
          vertices[ivertex].isGood() = true;
        }
      }
    }  // end for
    alpaka::syncBlockThreads(acc);
    for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize,alpaka::warp::getSize(acc)))) {
      if (itrackO < trackBlockSize){
        int itrack = itrackO + blockIdx * trackBlockSize;
        // Tracks are associated to vertex in list kmin, kmin+1,... kmax-1, so this just assign all tracks to the vertex we just created
        tracks[itrack].kmin() = maxVerticesPerBlock * blockIdx;
        tracks[itrack].kmax() = maxVerticesPerBlock * blockIdx + 1;
      }
    }
    alpaka::syncBlockThreads(acc);
  }

  ALPAKA_FN_ACC static void getBeta0(const Acc1D& acc,
                                     reco::TrackForVertexDeviceCollection::View tracks,
                                     reco::VertexDeviceCollection::View vertices,
                                     DAInBlocksClusterParameters const cParams,
                                     double& beta,
                                     int trackBlockSize) {
    // Computes first critical temperature
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize,alpaka::warp::getSize(acc)))) {
      if (itrackO < trackBlockSize){
        int itrack = itrackO + blockIdx * trackBlockSize;
        if (not(tracks[itrack].isGood()))
          continue;
        tracks[itrack].aux1() = tracks[itrack].weight() * tracks[itrack].oneoverdz2();
        tracks[itrack].aux2() = tracks[itrack].weight() * tracks[itrack].oneoverdz2() * tracks[itrack].z();
      }
    }
    // Initial vertex position
    alpaka::syncBlockThreads(acc);
    float& wnew = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& znew = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    if (once_per_block(acc)) {
      wnew = 0.;
      znew = 0.;
    }
    alpaka::syncBlockThreads(acc);
    for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize,alpaka::warp::getSize(acc)))) {
      if (itrackO < trackBlockSize){
        int itrack = itrackO + blockIdx * trackBlockSize;
        if (not(tracks[itrack].isGood()))
          continue;
        alpaka::atomicAdd(acc, &wnew, tracks[itrack].aux1(), alpaka::hierarchy::Threads{});
        alpaka::atomicAdd(acc, &znew, tracks[itrack].aux2(), alpaka::hierarchy::Threads{});
      }
    }
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)) {
      vertices[maxVerticesPerBlock * blockIdx].z() = znew / wnew;
      znew = 0.;
    }
    alpaka::syncBlockThreads(acc);
    // Now do a chi-2 like of all tracks and save it again in znew
    for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize,alpaka::warp::getSize(acc)))) {
      if (itrackO < trackBlockSize){
        int itrack = itrackO + blockIdx * trackBlockSize;
        if (not(tracks[itrack].isGood()))
          continue;
        tracks[itrack].aux2() =
            tracks[itrack].aux1() * (vertices[maxVerticesPerBlock * blockIdx].z() - tracks[itrack].z()) *
            (vertices[maxVerticesPerBlock * blockIdx].z() - tracks[itrack].z()) * tracks[itrack].oneoverdz2();
        alpaka::atomicAdd(acc, &znew, tracks[itrack].aux2(), alpaka::hierarchy::Threads{});
      }
    }
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)) {
      beta = 2 * znew / wnew;  // 1/beta_C, or T_C
      if (beta > cParams.Tmin) {
        int coolingsteps =
            1 - int(alpaka::math::log(acc, beta / cParams.Tmin) /
                    alpaka::math::log(
                        acc, cParams.coolingFactor));  // A tricky conversion to round the number of cooling steps
        beta = alpaka::math::pow(acc, cParams.coolingFactor, coolingsteps) / cParams.Tmin;
      } else {
        beta = cParams.coolingFactor / cParams.Tmin;
      }
    }
    alpaka::syncBlockThreads(acc);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerInitializers_h
