#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"

#include "DataFormats/VertexSoA/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/TrackForVertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

//#ifndef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
//#define DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO 5
//#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  struct ClusterParameters {
    double Tmin;
    double Tpurge;
    double Tstop;
    double vertexSize;
    double coolingFactor;
    double d0CutOff;
    double dzCutOff;
    double uniquetrkweight;
    double uniquetrkminp;
    double zmerge;
    double zrange;
    int32_t convergence_mode;
    double delta_lowT;
    double delta_highT;
  };

  class DAInBlocksClusterizerAlgo {
  public:
    DAInBlocksClusterizerAlgo(Queue& queue, int32_t bSize);

    void clusterize(Queue& queue,
                    TrackForVertexDeviceCollection& inputTracks,
                    VertexDeviceCollection& deviceVertex,
                    ClusterParameters const& cParams,
                    int32_t nBlocks,
                    int32_t blockSize);  // Clusterization

    void resplit_tracks(Queue& queue,
                        TrackForVertexDeviceCollection& inputTracks,
                        VertexDeviceCollection& deviceVertex,
                        ClusterParameters const& cParams,
                        int32_t nBlocks,
                        int32_t blockSize);  // Clusterization

    void reject_outliers(Queue& queue,
                         TrackForVertexDeviceCollection& inputTracks,
                         VertexDeviceCollection& deviceVertex,
                         ClusterParameters const& cParams,
                         int32_t nBlocks,
                         int32_t blockSize);  // Clusterization
    void arbitrate(Queue& queue,
                   TrackForVertexDeviceCollection& inputTracks,
                   VertexDeviceCollection& deviceVertex,
                   ClusterParameters const& cParams,
                   int32_t nBlocks,
                   int32_t blockSize);  // Arbitration

  private:
    cms::alpakatools::device_buffer<Device, double[]> beta_;
    cms::alpakatools::device_buffer<Device, double[]> osumtkwt_;
  };

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void dump(const TAcc& acc, double& beta, VertexDeviceCollection::View vertices) {
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock =
        (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
                        acc)[0u];  // Max vertices size is 1024 over number of blocks in the kernel
    printf("[DAInBlocksClusterizerAlgo::dump()] Block Idx %i with nV %i at beta %1.8f \n",
           blockIdx,
           vertices[blockIdx].nV(),
           beta);
    for (int ivertexO = maxVerticesPerBlock * blockIdx;
         ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV();
         ivertexO += 1) {
      int ivertex = vertices[ivertexO].order();
      printf(
          "[DAInBlocksClusterizerAlgo::dump()] -- Block Idx %i, vertex %i in order %i: "
          "z=%1.5f,swE=%1.5f,sw=%1.5f,pk=%1.5f\n",
          blockIdx,
          ivertex,
          ivertexO,
          vertices[ivertex].z(),
          vertices[ivertex].swE(),
          vertices[ivertex].sw(),
          vertices[ivertex].rho());
    }
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void set_vtx_range(const TAcc& acc,
                                          TrackForVertexDeviceCollection::View tracks,
                                          VertexDeviceCollection::View vertices,
                                          ClusterParameters const& cParams,
                                          double& osumtkwt,
                                          double& beta,
                                          int trackBlockSize) {
    // These updates the range of vertices associated to each track through the kmin/kmax variables
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  // Thread number inside block
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];     // Block number inside grid
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
                                              acc)[0u];  // Max vertices size is 1024 over number of blocks in grid
    double zrange_min_ = 0.1;                            // Hard coded as in CPU version
    for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
         itrack += blockSize) {
      // Based on current temperature (regularization term) and track position uncertainty, only keep relevant vertices
      double zrange = std::max(cParams.zrange / sqrt((beta)*tracks[itrack].oneoverdz2()), zrange_min_);
      double zmin = tracks[itrack].z() - zrange;
      // First the lower bound
      int kmin = std::min((int)(maxVerticesPerBlock * blockIdx) + vertices.nV(blockIdx) - 1, tracks[itrack].kmin());
      // If the vertex position in z is bigger than the minimum, go down through all vertices position until finding one that is too far
      if (vertices[vertices[kmin].order()].z() > zmin) {
        // i.e., while we find another vertex within range that is before the previous initial step
        while ((kmin > maxVerticesPerBlock * blockIdx) && ((vertices[vertices[kmin - 1].order()].z()) > zmin)) {
          kmin--;
        }
      } else {
        while ((kmin < (maxVerticesPerBlock * blockIdx + (int)(vertices[blockIdx].nV()) - 1)) &&
               ((vertices[vertices[kmin].order()].z()) < zmin)) {
          kmin++;
        }
      }

      // And now do the same for the upper bound
      double zmax = tracks[itrack].z() + zrange;
      int kmax = std::max(0,
                          std::min(maxVerticesPerBlock * blockIdx + (int)(vertices[blockIdx].nV()) - 1,
                                   (int)(tracks[itrack].kmax()) - 1));
      // For corner cases in which we purged the first vertex, thus not properly updating kmax during purging
      while (vertices[kmax].order() == 9999)
        kmax++;
      if (vertices[vertices[kmax].order()].z() < zmax) {
        // As long as we have more vertex above kmax but within z range, we can add them to the collection, keep going
        while ((kmax < (maxVerticesPerBlock * blockIdx + (int)(vertices[blockIdx].nV()) - 1)) &&
               ((vertices[vertices[kmax + 1].order()].z()) < zmax)) {
          kmax++;
        }
      } else {
        while ((kmax > maxVerticesPerBlock * blockIdx) && (vertices[vertices[kmax].order()].z() > zmax)) {
          kmax--;
        }
      }
      if (kmin <= kmax) {
        tracks[itrack].kmin() = (int)kmin;
        tracks[itrack].kmax() = (int)kmax + 1;
      } else {  // Track goes to the most extreme cases if no associated one is found
        tracks[itrack].kmin() = (int)std::max(maxVerticesPerBlock * blockIdx, (int)std::min(kmin, kmax));
        tracks[itrack].kmax() = (int)std::min((maxVerticesPerBlock * blockIdx) + (int)vertices[blockIdx].nV(),
                                              (int)std::max(kmin, kmax) + 1);
      }
    }  //end for
    alpaka::syncBlockThreads(acc);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void update(const TAcc& acc,
                                   TrackForVertexDeviceCollection::View tracks,
                                   VertexDeviceCollection::View vertices,
                                   ClusterParameters const& cParams,
                                   double& osumtkwt,
                                   double& beta,
                                   double rho0,
                                   bool updateTc,
                                   int trackBlockSize) {
    // Main function that updates the annealing parameters on each T step, computes all partition functions and so on
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  // Thread number inside block
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];     // Block number inside grid
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
                                              acc)[0u];  // Max vertices size is 1024 over number of blocks in grid
    // Initial partition function, really only used on the outlier rejection step to penalize
    double Zinit = rho0 * exp(-(beta)*cParams.dzCutOff * cParams.dzCutOff);
    for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
         itrack += blockSize) {
      double botrack_dz2 = -(beta)*tracks[itrack].oneoverdz2();
      tracks[itrack].sum_Z() = Zinit;
      for (int ivertexO = tracks[itrack].kmin(); ivertexO < tracks[itrack].kmax(); ++ivertexO) {
        int ivertex = vertices[ivertexO].order();
        if (not(vertices[ivertex].isGood()))
          continue;
        // Z_t = sum_v pho_v * e^{-beta*(z_t-z_v)/dz^2}, partition function of the track
        tracks[itrack].sum_Z() +=
            vertices[ivertex].rho() * exp(botrack_dz2 * (tracks[itrack].z() - vertices[ivertex].z()) *
                                          (tracks[itrack].z() - vertices[ivertex].z()));
      }  //end vertex for
      if (not(std::isfinite(tracks[itrack].sum_Z())))
        tracks[itrack].sum_Z() = 0;
      if (tracks[itrack].sum_Z() > 1e-40) {  // If non-zero then the track has a non-trivial assignment to a vertex
        double sumw = tracks[itrack].weight() / (double)tracks[itrack].sum_Z();
        for (int ivertexO = tracks[itrack].kmin(); ivertexO < tracks[itrack].kmax(); ++ivertexO) {
          int ivertex = vertices[ivertexO].order();
          if (not(vertices[ivertex].isGood()))
            continue;
          double vert_exparg = botrack_dz2 * (tracks[itrack].z() - vertices[ivertex].z()) *
                               (tracks[itrack].z() - vertices[ivertex].z());  // -beta*(z_t-z_v)/dz^2
          double vert_exp = exp(vert_exparg);                                 // e^{-beta*(z_t-z_v)/dz^2}
          tracks[itrack].vert_se()[ivertex] =
              vert_exp * sumw;  // From partition of track to contribution of track to vertex partition
          double w = vertices[ivertex].rho() * vert_exp * sumw * tracks[itrack].oneoverdz2();
          tracks[itrack].vert_sw()[ivertex] = w;
          tracks[itrack].vert_swz()[ivertex] = w * tracks[itrack].z();
          if (updateTc) {
            tracks[itrack].vert_swE()[ivertex] = -w * vert_exparg / (beta);
          } else {
            tracks[itrack].vert_swE()[ivertex] = 0;
          }
        }  //end vertex for
      }  //end if
    }  //end track for
    alpaka::syncBlockThreads(acc);
    // After the track-vertex matrix assignment, we need to add up across vertices. This time, we use one thread per vertex
    for (int ivertexO = maxVerticesPerBlock * blockIdx + threadIdx;
         ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV();
         ivertexO += blockSize) {
      int ivertex = vertices[ivertexO].order();
      float se = 0.;
      float sw = 0.;
      float swz = 0.;
      float swE = 0.;
      for (int itrack = blockIdx * trackBlockSize; itrack < (blockIdx + 1) * trackBlockSize; itrack++) {
        if ((ivertexO >= tracks[itrack].kmin()) && (ivertexO < tracks[itrack].kmax())) {
          se += tracks[itrack].vert_se()[ivertex];
          sw += tracks[itrack].vert_sw()[ivertex];
          swz += tracks[itrack].vert_swz()[ivertex];
          swE += tracks[itrack].vert_swE()[ivertex];
        }
      }
      if (sw > 0) {
        double znew = swz / sw;
        vertices[ivertex].aux1() = znew - vertices[ivertex].z();
        vertices[ivertex].z() = znew;
      } else {
        vertices[ivertex].aux1() = 0.;
      }
      vertices[ivertex].rho() = vertices[ivertex].rho() * se * osumtkwt;
      vertices[ivertex].sw() = sw;
      if (updateTc)
        vertices[ivertex].swE() = swE;
    }  // end vertex for
    alpaka::syncBlockThreads(acc);
  }  //end update

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void merge(const TAcc& acc,
                                  TrackForVertexDeviceCollection::View tracks,
                                  VertexDeviceCollection::View vertices,
                                  ClusterParameters const& cParams,
                                  double& osumtkwt,
                                  double& beta,
                                  int trackBlockSize) {
    // If two vertex are too close together, merge them
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
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
    for (int ivertexO = maxVerticesPerBlock * blockIdx + threadIdx;
         ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV() - 1;
         ivertexO += blockSize) {
      int ivertex = vertices[ivertexO].order();
      int ivertexnext = vertices[ivertexO + 1].order();
      vertices[ivertex].aux1() = abs(vertices[ivertex].z() - vertices[ivertexnext].z());
    }
    alpaka::syncBlockThreads(acc);
    // Sorter things
    auto& critical_dist = alpaka::declareSharedVar<float[128], __COUNTER__>(acc);
    auto& critical_index = alpaka::declareSharedVar<float[128], __COUNTER__>(acc);
    int& ncritical = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    if (once_per_block(acc)) {
      ncritical = 0;
      for (int ivertexO = maxVerticesPerBlock * blockIdx + threadIdx;
           ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV() - 1;
           ivertexO += blockSize) {
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
    for (int sortO = 0; sortO < ncritical; ++sortO) {
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
      critical_dist[ikO] = 9999999.;
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
      for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
           itrack += blockSize) {
        if (tracks[itrack].kmax() > ivertexO + 1)
          tracks[itrack].kmax()--;
        if ((tracks[itrack].kmin() > ivertexO) || ((tracks[itrack].kmax() < (tracks[itrack].kmin() + 1)) &&
                                                   (tracks[itrack].kmin() > maxVerticesPerBlock * blockIdx)))
          tracks[itrack].kmin()--;
      }
      alpaka::syncBlockThreads(acc);
      set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
      return;
    }
    alpaka::syncBlockThreads(acc);
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void split(const TAcc& acc,
                                  TrackForVertexDeviceCollection::View tracks,
                                  VertexDeviceCollection::View vertices,
                                  ClusterParameters const& cParams,
                                  double& osumtkwt,
                                  double& beta,
                                  double threshold,
                                  int trackBlockSize) {
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    update(acc, tracks, vertices, cParams, osumtkwt, beta, 0.0, true, trackBlockSize);
    double epsilon = 1e-3;
    int nprev = vertices[blockIdx].nV();
    // Set critical T for all vertices
    for (int ivertexO = maxVerticesPerBlock * blockIdx + threadIdx;
         ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV();
         ivertexO += blockSize) {
      int ivertex = vertices[ivertexO].order();
      double Tc = 2 * vertices[ivertex].swE() / vertices[ivertex].sw();
      vertices[ivertex].aux1() = Tc;
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
      for (int ivertexO = maxVerticesPerBlock * blockIdx + threadIdx;
           ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV();
           ivertexO += blockSize) {
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
      for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
           itrack += blockSize) {
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
      int nnew = 999999;
      // Find the first empty index to save the vertex
      for (int icheck = maxVerticesPerBlock * blockIdx; icheck < maxVerticesPerBlock * (blockIdx + 1); icheck++) {
        if (not(vertices[icheck].isGood())) {
          nnew = icheck;
          break;
        }
      }
      if (nnew == 999999)
        break;  // Need to compute in all threads so all exit properly
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
      for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
           itrack += blockSize) {
        if (tracks[itrack].kmin() > ivertexO)
          tracks[itrack].kmin()++;
        if ((tracks[itrack].kmax() >= ivertexO) || (tracks[itrack].kmax() == tracks[itrack].kmin()))
          tracks[itrack].kmax()++;
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

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void purge(const TAcc& acc,
                                  TrackForVertexDeviceCollection::View tracks,
                                  VertexDeviceCollection::View vertices,
                                  ClusterParameters const& cParams,
                                  double& osumtkwt,
                                  double& beta,
                                  double rho0,
                                  int trackBlockSize) {
    // Remove repetitive or low quality entries
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
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
    for (int ivertexO = maxVerticesPerBlock * blockIdx + threadIdx;
         ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV();
         ivertexO += blockSize) {
      int ivertex = vertices[ivertexO].order();
      vertices[ivertex].aux1() = 0;  // sum of track-vertex probabilities
      vertices[ivertex].aux2() = 0;  // number of uniquely assigned tracks
    }
    alpaka::syncBlockThreads(acc);
    // Get quality of vertex in terms of #Tracks and sum of track probabilities
    for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
         itrack += blockSize) {
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
    alpaka::syncBlockThreads(acc);
    // Find worst vertex to purge
    int& k0 = alpaka::declareSharedVar<int, __COUNTER__>(acc);

    if (once_per_block(acc)) {
      double sumpmin = tracks.nT();
      k0 = maxVerticesPerBlock * blockIdx + nprev;
      for (int ivertexO = maxVerticesPerBlock * blockIdx + threadIdx;
           ivertexO < maxVerticesPerBlock * blockIdx + (int)vertices[blockIdx].nV();
           ivertexO += blockSize) {
        int ivertex = vertices[ivertexO].order();
        if ((vertices[ivertex].aux2() < nunique_min) && (vertices[ivertex].aux1() < sumpmin)) {
          sumpmin = vertices[ivertex].aux1();
          k0 = ivertexO;
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
      for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
           itrack += blockSize) {
        if (tracks[itrack].kmax() > k0)
          tracks[itrack].kmax()--;
        if ((tracks[itrack].kmin() > k0) || ((tracks[itrack].kmax() < (tracks[itrack].kmin() + 1)) &&
                                             (tracks[itrack].kmin() > (int)(maxVerticesPerBlock * blockIdx))))
          tracks[itrack].kmin()--;
      }
    }  // end if
    alpaka::syncBlockThreads(acc);
    if (nprev != vertices[blockIdx].nV()) {
      set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    }
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void initialize(const TAcc& acc,
                                       TrackForVertexDeviceCollection::View tracks,
                                       VertexDeviceCollection::View vertices,
                                       ClusterParameters const& cParams,
                                       int trackBlockSize) {
    // Initialize all vertices as empty, a single vertex in each block will be initialized with all tracks associated to it
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    vertices[blockIdx].nV() = 1;
    for (int ivertex = threadIdx + maxVerticesPerBlock * blockIdx; ivertex < maxVerticesPerBlock * (blockIdx + 1);
         ivertex +=
         blockSize) {  // Initialize vertices in parallel in the block. Note that a block of threads should always be restricted to oeprations within maxVerticesPerBlock*blockIdx and maxVerticesPerBlock*(blockIdx+1)-1 to avoid running conditions
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
    }  // end for
    alpaka::syncBlockThreads(acc);
    for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
         itrack += blockSize) {
      // Tracks are associated to vertex in list kmin, kmin+1,... kmax-1, so this just assign all tracks to the vertex we just created
      tracks[itrack].kmin() = maxVerticesPerBlock * blockIdx;
      tracks[itrack].kmax() = maxVerticesPerBlock * blockIdx + 1;
    }
    alpaka::syncBlockThreads(acc);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void getBeta0(const TAcc& acc,
                                     TrackForVertexDeviceCollection::View tracks,
                                     VertexDeviceCollection::View vertices,
                                     ClusterParameters const& cParams,
                                     double& beta,
                                     int trackBlockSize) {
    // Computes first critical temperature
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
         itrack += blockSize) {
      if (not(tracks[itrack].isGood()))
        continue;
      tracks[itrack].aux1() = tracks[itrack].weight() * tracks[itrack].oneoverdz2();
      tracks[itrack].aux2() = tracks[itrack].weight() * tracks[itrack].oneoverdz2() * tracks[itrack].z();
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
    for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
         itrack += blockSize) {
      if (not(tracks[itrack].isGood()))
        continue;
      alpaka::atomicAdd(acc, &wnew, tracks[itrack].aux1(), alpaka::hierarchy::Threads{});
      alpaka::atomicAdd(acc, &znew, tracks[itrack].aux2(), alpaka::hierarchy::Threads{});
    }
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)) {
      vertices[maxVerticesPerBlock * blockIdx].z() = znew / wnew;
      znew = 0.;
    }
    alpaka::syncBlockThreads(acc);
    // Now do a chi-2 like of all tracks and save it again in znew
    for (int itrack = threadIdx + blockIdx * trackBlockSize; itrack < threadIdx + (blockIdx + 1) * trackBlockSize;
         itrack += blockSize) {
      if (not(tracks[itrack].isGood()))
        continue;
      tracks[itrack].aux2() =
          tracks[itrack].aux1() * (vertices[maxVerticesPerBlock * blockIdx].z() - tracks[itrack].z()) *
          (vertices[maxVerticesPerBlock * blockIdx].z() - tracks[itrack].z()) * tracks[itrack].oneoverdz2();
      alpaka::atomicAdd(acc, &znew, tracks[itrack].aux2(), alpaka::hierarchy::Threads{});
    }
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)) {
      beta = 2 * znew / wnew;  // 1/beta_C, or T_C
      if (beta > cParams.Tmin) {
        int coolingsteps =
            1 - int(std::log(beta / cParams.Tmin) /
                    std::log(cParams.coolingFactor));  // A tricky conversion to round the number of cooling steps
        beta = std::pow(cParams.coolingFactor, coolingsteps) / cParams.Tmin;
      } else {
        beta = cParams.coolingFactor / cParams.Tmin;
      }
    }
    alpaka::syncBlockThreads(acc);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void thermalize(const TAcc& acc,
                                       TrackForVertexDeviceCollection::View tracks,
                                       VertexDeviceCollection::View vertices,
                                       ClusterParameters const& cParams,
                                       double& osumtkwt,
                                       double& beta,
                                       double delta_highT,
                                       double rho0,
                                       int trackBlockSize) {
    // At a fixed temperature, iterate vertex position update until stable
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int niter = 0;
    double zrange_min_ = 0.01;  // Hard coded as it is in CPU
    double delta_max = cParams.delta_lowT;
    if (cParams.convergence_mode == 0) {
      delta_max = delta_highT;
    } else if (cParams.convergence_mode == 1) {
      delta_max = cParams.delta_lowT / sqrt(std::max(beta, 1.0));
    }
    int maxIterations = 1000;
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
    // Accumulator of variations
    double delta_sum_range = 0;
    while (niter++ < maxIterations) {
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgo::thermalize()] BlockIdx %i, thermalize at beta=%1.3f, iteration %i\n",
               blockIdx,
               beta,
               niter);
      }
#endif
      update(acc, tracks, vertices, cParams, osumtkwt, beta, rho0, false, trackBlockSize);
      double dmax = 0.;
      for (int ivertexO = maxVerticesPerBlock * blockIdx;
           ivertexO < maxVerticesPerBlock * blockIdx + vertices[blockIdx].nV();
           ivertexO++) {
        int ivertex = vertices[ivertexO].order();
        if (vertices[ivertex].aux1() >= dmax)
          dmax = vertices[ivertex].aux1();
      }
      delta_sum_range += dmax;
      // If a vertex moved too much we reassign
      if (delta_sum_range > zrange_min_ && dmax > zrange_min_) {
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
        if (once_per_block(acc)) {
          printf(
              "[DAInBlocksClusterizerAlgo::thermalize()] BlockIdx %i, thermalize at beta=%1.3f, iteration %i. Found "
              "delta_sum_range=%1.3f, dmax=%1.3f, will redo track-vertex assignament\n",
              blockIdx,
              beta,
              niter,
              delta_sum_range,
              dmax);
        }
#endif
        set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
        delta_sum_range = 0.;
      }
      if (dmax < delta_max) {
        update(acc, tracks, vertices, cParams, osumtkwt, beta, 0.0, true, trackBlockSize);

#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
        if (once_per_block(acc)) {
          printf(
              "[DAInBlocksClusterizerAlgo::thermalize()] BlockIdx %i, thermalize at beta=%1.3f, iteration %i. Found "
              "delta_sum_range=%1.3f, dmax=%1.3f, all vertices stable enough to stop thermalizing\n",
              blockIdx,
              beta,
              niter,
              delta_sum_range,
              dmax);
          dump(acc, beta, vertices);
        }
#endif
        break;
      }
    }  // end while
  }  // thermalize

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void coolingWhileSplitting(const TAcc& acc,
                                                  TrackForVertexDeviceCollection::View tracks,
                                                  VertexDeviceCollection::View vertices,
                                                  ClusterParameters const& cParams,
                                                  double& osumtkwt,
                                                  double& beta,
                                                  int trackBlockSize) {
    // Perform cooling of the deterministic annealing
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    double betafreeze = (1. / cParams.Tmin) * sqrt(cParams.coolingFactor);
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

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void reMergeTracks(const TAcc& acc,
                                          TrackForVertexDeviceCollection::View tracks,
                                          VertexDeviceCollection::View vertices,
                                          ClusterParameters const& cParams,
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

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void reSplitTracks(const TAcc& acc,
                                          TrackForVertexDeviceCollection::View tracks,
                                          VertexDeviceCollection::View vertices,
                                          ClusterParameters const& cParams,
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

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC static void rejectOutliers(const TAcc& acc,
                                           TrackForVertexDeviceCollection::View tracks,
                                           VertexDeviceCollection::View vertices,
                                           ClusterParameters const& cParams,
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
        beta = std::min(beta / cParams.coolingFactor, 1. / cParams.Tpurge);
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
        beta = std::min(beta / cParams.coolingFactor, 1. / cParams.Tstop);
      }
      alpaka::syncBlockThreads(acc);
      thermalize(acc, tracks, vertices, cParams, osumtkwt, beta, cParams.delta_lowT, rho0, trackBlockSize);
    }
    alpaka::syncBlockThreads(acc);
    // The last track to vertex assignment!
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, beta, trackBlockSize);
  }  // rejectOutliers

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerAlgo_h
