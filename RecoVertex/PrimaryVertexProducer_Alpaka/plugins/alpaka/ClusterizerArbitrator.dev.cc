#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"

#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/ClusterizerAlgo.h"

#define DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_ARBITRATOR 1

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC static void resortVerticesAndAssign(const TAcc& acc,
                                                    portablevertex::TrackDeviceCollection::View tracks,
                                                    portablevertex::VertexDeviceCollection::View vertices,
                                                    const portablevertex::ClusterParamsHostCollection::ConstView cParams,
                                                    int32_t griddim) {
    // Multiblock vertex arbitration
    double beta = 1. / cParams.Tstop();
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  // Thread number inside block
    auto& z = alpaka::declareSharedVar<float[128], __COUNTER__>(acc);
    auto& rho = alpaka::declareSharedVar<float[128], __COUNTER__>(acc);
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)) {
      int nTrueVertex = 0;
      int maxVerticesPerBlock = (int)512 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
                                               acc)[0u];  // Max vertices size is 512 over number of blocks in grid
      for (int32_t blockid = 0; blockid < griddim; blockid++) {
        for (int ivtx = blockid * maxVerticesPerBlock; ivtx < blockid * maxVerticesPerBlock + vertices[blockid].nV();
             ivtx++) {
          int ivertex = vertices[ivtx].order();
          if ((vertices[ivertex].rho() < 10000) && (abs(vertices[ivertex].z()) < 30)) {
            z[nTrueVertex] = vertices[ivertex].z();
            rho[nTrueVertex] = vertices[ivertex].rho();
            nTrueVertex++;
            if (nTrueVertex == 1024)
              break;
          }
        }
      }
      vertices[0].nV() = nTrueVertex;
    }
    alpaka::syncBlockThreads(acc);

    auto& orderedIndices = alpaka::declareSharedVar<uint16_t[1024], __COUNTER__>(acc);
    auto& sws = alpaka::declareSharedVar<uint16_t[1024], __COUNTER__>(acc);

    int const& nvFinal = vertices[0].nV();

    cms::alpakatools::radixSort<Acc1D, float, 2>(acc, z, orderedIndices, sws, nvFinal);
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)) {
      // copy sorted vertices back to the SoA
      for (int ivtx = threadIdx; ivtx < vertices[0].nV(); ivtx += blockSize) {
        vertices[ivtx].z() = z[ivtx];
        vertices[ivtx].rho() = rho[ivtx];
        vertices[ivtx].order() = orderedIndices[ivtx];
      }
    }
    alpaka::syncBlockThreads(acc);
    double zrange_min_ = 0.1;

    for (int itrack = threadIdx; itrack < tracks.nT(); itrack += blockSize) {
      if (not(tracks[itrack].isGood()))
        continue;
      double zrange = std::max(cParams.zrange() / sqrt((beta)*tracks[itrack].oneoverdz2()), zrange_min_);
      double zmin = tracks[itrack].z() - zrange;
      int kmin = vertices[0].nV() - 1;
      if (kmin < 0)
        continue;
      if (vertices[vertices[kmin].order()].z() > zmin) {  // vertex properties always accessed through vertices->order
        while ((kmin > 0) &&
               (vertices[vertices[kmin - 1].order()].z() >
                zmin)) {  // i.e., while we find another vertex within range that is before the previous initial step
          kmin--;
        }
      } else {
        while ((kmin < vertices[0].nV()) &&
               (vertices[vertices[kmin].order()].z() <
                zmin)) {  // Or it might happen that we have to take out vertices from the thing
          kmin++;
        }
      }
      // Now the same for the upper bound
      double zmax = tracks[itrack].z() + zrange;
      int kmax = 0;
      if (vertices[vertices[kmax].order()].z() < zmax) {
        while (
            (kmax < vertices[0].nV() - 1) &&
            (vertices[vertices[kmax + 1].order()].z() <
             zmax)) {  // As long as we have more vertex above kmax but within z range, we can add them to the collection, keep going
          kmax++;
        }
      } else {  //Or maybe we have to restrict it
        while ((kmax > 0) && (vertices[vertices[kmax].order()].z() > zmax)) {
          kmax--;
        }
      }
      if (kmin <= kmax) {
        tracks[itrack].kmin() = kmin;
        tracks[itrack].kmax() = kmax + 1;  //always looping to tracks->kmax(i) - 1
      } else {                             // If it is here, the whole vertex are under
        tracks[itrack].kmin() = std::max(0, std::min(kmin, kmax));
        tracks[itrack].kmax() = std::min(vertices[0].nV(), std::max(kmin, kmax) + 1);
      }
    }
    alpaka::syncBlockThreads(acc);

    double mintrkweight_ = 0.5;
    double rho0 = vertices[0].nV() > 1 ? 1. / vertices[0].nV() : 1.;
    double z_sum_init = rho0 * exp(-(beta)*cParams.dzCutOff() * cParams.dzCutOff());
    for (int itrack = threadIdx; itrack < tracks.nT(); itrack += blockSize) {
      int kmin = tracks[itrack].kmin();
      int kmax = tracks[itrack].kmax();
      double p_max = -1;
      int iMax = 10000;
      double sum_Z = z_sum_init;
      for (auto k = kmin; k < kmax; k++) {
        double v_exp = exp(-(beta)*std::pow(tracks[itrack].z() - vertices[vertices[k].order()].z(), 2) *
                           tracks[itrack].oneoverdz2());
        sum_Z += vertices[vertices[k].order()].rho() * v_exp;
      }
      double invZ = sum_Z > 1e-100 ? 1. / sum_Z : 0.0;
      for (auto k = kmin; k < kmax; k++) {
        float v_exp = exp(-(beta)*std::pow(tracks[itrack].z() - vertices[vertices[k].order()].z(), 2) *
                          tracks[itrack].oneoverdz2());
        float p = vertices[vertices[k].order()].rho() * v_exp * invZ;
        if (p > p_max && p > mintrkweight_) {
          // assign  track i -> vertex k (hard, mintrkweight_ should be >= 0.5 here)
          p_max = p;
          iMax = k;
        }
      }
      tracks[itrack].kmin() = iMax;
      tracks[itrack].kmax() = iMax + 1;
    }
    alpaka::syncBlockThreads(acc);
  }  //resortVerticesAndAssign

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC static void finalizeVertices(const TAcc& acc,
                                             portablevertex::TrackDeviceCollection::View tracks,
                                             portablevertex::VertexDeviceCollection::View vertices,
                                             const portablevertex::ClusterParamsHostCollection::ConstView cParams) {
    //int blockSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    //int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; // Thread number inside block
    // From here it used to be vertices
    if (once_per_block(acc)) {
      for (int k = 0; k < vertices[0].nV(); k += 1) {  //TODO: ithread, blockSize
        int ivertex = vertices[k].order();
        vertices[ivertex].ntracks() = 0;
        for (int itrack = 0; itrack < tracks.nT(); itrack += 1) {
          if (not(tracks[itrack].isGood()))
            continue;  // Remove duplicates
          int ivtxFromTk = tracks[itrack].kmin();
          if (ivtxFromTk == k) {
            bool isNew = true;
            for (int ivtrack = 0; ivtrack < vertices[ivertex].ntracks(); ivtrack++) {
              if (tracks[itrack].tt_index() == tracks[vertices[ivertex].track_id()[ivtrack]].tt_index())
                isNew = false;
            }
            if (!isNew)
              continue;
            vertices[ivertex].track_id()[vertices[ivertex].ntracks()] = itrack;  //tracks[itrack].tt_index();
            vertices[ivertex].track_weight()[vertices[ivertex].ntracks()] = 1.;
            vertices[ivertex].ntracks()++;
          }
        }
        if (vertices[ivertex].ntracks() < 2) {
          vertices[ivertex].isGood() = false;  // No longer needed
          continue;                            //Skip vertex if it has no tracks
        }
        vertices[ivertex].x() = 0;
        vertices[ivertex].y() = 0;
      }
    }
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)) {
      // So we now check whether each vertex is further enough from the previous one
      for (int k = 0; k < vertices[0].nV(); k++) {
        int prevVertex = ((int)k) - 1;
        int thisVertex = (int)vertices[k].order();
        if (not(vertices[thisVertex].isGood())) {
          continue;
        }
        while (prevVertex >= 0) {
          // Find the previous vertex that was good
          if (!vertices[vertices[prevVertex].order()].isGood())
            break;  //Can't be part of the while condition as otherwise it could try to look up with index -1 in the vervex view
          prevVertex--;
        }
        if ((prevVertex < 0)) {  // If it is first, always good
          vertices[thisVertex].isGood() = true;
        } else if (abs(vertices[thisVertex].z() - vertices[prevVertex].z()) >
                   (2 * cParams.vertexSize())) {  //If it is further away enough, it is also good
          vertices[thisVertex].isGood() = true;
        } else {
          vertices[thisVertex].isGood() = false;
        }
      }
      // We have to deal with the order being broken by the invalidation of vertexes and set back again the vertex multiplicity, unfortunately can't be parallelized without threads competing
      int k = 0;
      while (k != vertices[0].nV()) {
        int thisVertex = vertices[k].order();
        if (vertices[thisVertex].isGood()) {  // If is good just continue
          k++;
        } else {
          for (int l = k; l < vertices[0].nV(); l++) {  //If it is bad, move one position all indexes
            vertices[l].order() = vertices[l + 1].order();
          }
          vertices[0].nV()--;  // And reduce vertex number by 1
        }
      }
    }
    alpaka::syncBlockThreads(acc);
  }  //finalizeVertices

  class arbitrateKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  portablevertex::TrackDeviceCollection::View tracks,
                                  portablevertex::VertexDeviceCollection::View vertices,
                                  const portablevertex::ClusterParamsHostCollection::ConstView cParams,
                                  int32_t nBlocks) const {
      // This has the core of the clusterization algorithm
      int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];  // Block number inside grid
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgoArbitrator::operator()] Start arbitration for block %i\n", blockIdx);
      }
#endif
      resortVerticesAndAssign(acc, tracks, vertices, cParams, nBlocks);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgoArbitrator::operator()] Vertex reassignment finished for block %i\n", blockIdx);
      }
#endif
      alpaka::syncBlockThreads(acc);
      finalizeVertices(acc, tracks, vertices, cParams);  // In CUDA it used to be verticesAndClusterize
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[ClusterizerAlgoArbitrator::operator()] Vertices finalized for block %i\n", blockIdx);
      }
#endif
      alpaka::syncBlockThreads(acc);
    }
  };  // class kernel

  void ClusterizerAlgo::arbitrate(Queue& queue,
                                  portablevertex::TrackDeviceCollection& deviceTrack,
                                  portablevertex::VertexDeviceCollection& deviceVertex,
                                  const std::shared_ptr<portablevertex::ClusterParamsHostCollection> cParams,
                                  int32_t nBlocks,
                                  int32_t blockSize) {
    const int blocks = divide_up_by(blockSize, blockSize);  //Single block, as it has to converge to a single collection
    alpaka::exec<Acc1D>(
        queue,
        make_workdiv<Acc1D>(blocks, blockSize),
        arbitrateKernel{},
        deviceTrack
            .view(),  // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
        deviceVertex.view(),
        cParams->view(),
        nBlocks);
  }  // arbitraterAlgo::arbitrate

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
