#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerPrimitives_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerPrimitives_h

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

#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_CLUSTERIZERALGO
  ALPAKA_FN_ACC static void dump(const Acc1D& acc, double& beta, reco::VertexDeviceCollection::View vertices) {
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int maxVerticesPerBlock =
        (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
                        acc)[0u];  // Max vertices size is 1024 over number of blocks in the kernel
    printf("[DAInBlocksClusterizerAlgo::dump()] Block Idx %i with nV %i at beta %1.8f \n",
           blockIdx,
           vertices[blockIdx].nV(),
           beta);
    for (int ivertexO : uniform_elements(acc, round_up_by(vertices[blockIdx].nV(), alpaka::warp::getSize(acc)))) {
      if (ivertexO < vertices[blockIdx].nV()) {
        int ivertex = vertices[maxVerticesPerBlock * blockIdx + ivertexO].order();
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
  }
#endif
  ALPAKA_FN_ACC static void set_vtx_range(const Acc1D& acc,
                                          reco::TrackForVertexDeviceCollection::View tracks,
                                          reco::VertexDeviceCollection::View vertices,
                                          DAInBlocksClusterParameters const cParams,
                                          double& osumtkwt,
                                          double& beta,
                                          int trackBlockSize) {
    // These updates the range of vertices associated to each track through the kmin/kmax variables
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];  // Block number inside grid
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
                                              acc)[0u];  // Max vertices size is 1024 over number of blocks in grid
    double zrange_min = 0.1;                             // Hard coded as in CPU version
    for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize, alpaka::warp::getSize(acc)))) {
      if (itrackO < trackBlockSize) {
        int itrack = itrackO + blockIdx * trackBlockSize;
        // Based on current temperature (regularization term) and track position uncertainty, only keep relevant vertices
        double zrange = alpaka::math::max(
            acc, cParams.zrange / alpaka::math::sqrt(acc, (beta)*tracks[itrack].oneoverdz2()), zrange_min);
        double zmin = tracks[itrack].z() - zrange;
        // First the lower bound
        int kmin = alpaka::math::min(
            acc, (int)(maxVerticesPerBlock * blockIdx) + vertices.nV(blockIdx) - 1, tracks[itrack].kmin());
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
        int kmax =
            alpaka::math::max(acc,
                              0,
                              alpaka::math::min(acc,
                                                maxVerticesPerBlock * blockIdx + (int)(vertices[blockIdx].nV()) - 1,
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
          tracks[itrack].kmin() =
              (int)alpaka::math::max(acc, maxVerticesPerBlock * blockIdx, (int)alpaka::math::min(acc, kmin, kmax));
          tracks[itrack].kmax() =
              (int)alpaka::math::min(acc,
                                     (maxVerticesPerBlock * blockIdx) + (int)vertices[blockIdx].nV(),
                                     (int)alpaka::math::max(acc, kmin, kmax) + 1);
        }
      }  //end for
    }
    alpaka::syncBlockThreads(acc);
  }

  ALPAKA_FN_ACC static void update(const Acc1D& acc,
                                   reco::TrackForVertexDeviceCollection::View tracks,
                                   reco::VertexDeviceCollection::View vertices,
                                   DAInBlocksClusterParameters const cParams,
                                   double& osumtkwt,
                                   double& beta,
                                   double rho0,
                                   bool updateTc,
                                   int trackBlockSize) {
    // Main function that updates the annealing parameters on each T step, computes all partition functions and so on
    int blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];  // Block number inside grid
    int maxVerticesPerBlock = (int)1024 / alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
                                              acc)[0u];  // Max vertices size is 1024 over number of blocks in grid
    // Initial partition function, really only used on the outlier rejection step to penalize
    double Zinit = rho0 * alpaka::math::exp(acc, -(beta)*cParams.dzCutOff * cParams.dzCutOff);
    for (int itrackO : uniform_elements(acc, round_up_by(trackBlockSize, alpaka::warp::getSize(acc)))) {
      if (itrackO < trackBlockSize) {
        int itrack = itrackO + blockIdx * trackBlockSize;
        double botrack_dz2 = -(beta)*tracks[itrack].oneoverdz2();
        tracks[itrack].sum_Z() = Zinit;
        for (int ivertexO = tracks[itrack].kmin(); ivertexO < tracks[itrack].kmax(); ++ivertexO) {
          int ivertex = vertices[ivertexO].order();
          if (not(vertices[ivertex].isGood()))
            continue;
          // Z_t = sum_v pho_v * e^{-beta*(z_t-z_v)/dz^2}, partition function of the track
          tracks[itrack].sum_Z() +=
              vertices[ivertex].rho() * alpaka::math::exp(acc,
                                                          botrack_dz2 * (tracks[itrack].z() - vertices[ivertex].z()) *
                                                              (tracks[itrack].z() - vertices[ivertex].z()));
        }  //end vertex for
        if (not(alpaka::math::isfinite(acc, tracks[itrack].sum_Z())))
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
      }
    }  //end track for
    alpaka::syncBlockThreads(acc);
    // After the track-vertex matrix assignment, we need to add up across vertices. This time, we use one thread per vertex
    for (int ivertexO : uniform_elements(acc, round_up_by(vertices[blockIdx].nV(), alpaka::warp::getSize(acc)))) {
      if (ivertexO < vertices[blockIdx].nV()) {
        int ivertexC = maxVerticesPerBlock * blockIdx + ivertexO;
        int ivertex = vertices[ivertexC].order();
        float se = 0.;
        float sw = 0.;
        float swz = 0.;
        float swE = 0.;
        for (int itrack = blockIdx * trackBlockSize; itrack < (blockIdx + 1) * trackBlockSize; itrack++) {
          if ((ivertexC >= tracks[itrack].kmin()) && (ivertexC < tracks[itrack].kmax())) {
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
      }
    }  // end vertex for
    alpaka::syncBlockThreads(acc);
  }  //end update

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_DAInBlocksClusterizerPrimitives_h
