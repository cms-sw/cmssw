#ifndef RecoTracker_PixelVertexFinding_plugins_gpuClusterTracksIterative_h
#define RecoTracker_PixelVertexFinding_plugins_gpuClusterTracksIterative_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  // this algo does not really scale as it works in a single block...
  // enough for <10K tracks we have
  __global__ void clusterTracksIterative(VtxSoAView pdata,
                                         WsSoAView pws,
                                         int minT,      // min number of neighbours to be "core"
                                         float eps,     // max absolute distance to cluster
                                         float errmax,  // max error to be "seed"
                                         float chi2max  // max normalized distance to cluster
  ) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    if (verbose && 0 == threadIdx.x)
      printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);

    auto er2mx = errmax * errmax;

    auto& __restrict__ data = pdata;
    auto& __restrict__ ws = pws;
    auto nt = ws.ntrks();
    float const* __restrict__ zt = ws.zt();
    float const* __restrict__ ezt2 = ws.ezt2();

    uint32_t& nvFinal = data.nvFinal();
    uint32_t& nvIntermediate = ws.nvIntermediate();

    uint8_t* __restrict__ izt = ws.izt();
    int32_t* __restrict__ nn = data.ndof();
    int32_t* __restrict__ iv = ws.iv();

    assert(zt);
    assert(nn);
    assert(iv);
    assert(ezt2);

    using Hist = cms::cuda::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;
    __shared__ Hist hist;
    __shared__ typename Hist::Counter hws[32];
    for (auto j = threadIdx.x; j < Hist::totbins(); j += blockDim.x) {
      hist.off[j] = 0;
    }
    __syncthreads();

    if (verbose && 0 == threadIdx.x)
      printf("booked hist with %d bins, size %d for %d tracks\n", hist.nbins(), hist.capacity(), nt);

    assert((int)nt <= hist.capacity());

    // fill hist  (bin shall be wider than "eps")
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      assert(i < zVertex::utilities::MAXTRACKS);
      int iz = int(zt[i] * 10.);  // valid if eps<=0.1
      iz = std::clamp(iz, INT8_MIN, INT8_MAX);
      izt[i] = iz - INT8_MIN;
      assert(iz - INT8_MIN >= 0);
      assert(iz - INT8_MIN < 256);
      hist.count(izt[i]);
      iv[i] = i;
      nn[i] = 0;
    }
    __syncthreads();
    if (threadIdx.x < 32)
      hws[threadIdx.x] = 0;  // used by prefix scan...
    __syncthreads();
    hist.finalize(hws);
    __syncthreads();
    assert(hist.size() == nt);
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      hist.fill(izt[i], uint16_t(i));
    }
    __syncthreads();

    // count neighbours
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (ezt2[i] > er2mx)
        continue;
      auto loop = [&](uint32_t j) {
        if (i == j)
          return;
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > eps)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;
        nn[i]++;
      };

      cms::cuda::forEachInBins(hist, izt[i], 1, loop);
    }

    __shared__ int nloops;
    nloops = 0;

    __syncthreads();

    // cluster seeds only
    bool more = true;
    while (__syncthreads_or(more)) {
      if (1 == nloops % 2) {
        for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
          auto m = iv[i];
          while (m != iv[m])
            m = iv[m];
          iv[i] = m;
        }
      } else {
        more = false;
        for (auto k = threadIdx.x; k < hist.size(); k += blockDim.x) {
          auto p = hist.begin() + k;
          auto i = (*p);
          auto be = std::min(Hist::bin(izt[i]) + 1, int(hist.nbins() - 1));
          if (nn[i] < minT)
            continue;  // DBSCAN core rule
          auto loop = [&](uint32_t j) {
            assert(i != j);
            if (nn[j] < minT)
              return;  // DBSCAN core rule
            auto dist = std::abs(zt[i] - zt[j]);
            if (dist > eps)
              return;
            if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
              return;
            auto old = atomicMin(&iv[j], iv[i]);
            if (old != iv[i]) {
              // end the loop only if no changes were applied
              more = true;
            }
            atomicMin(&iv[i], old);
          };
          ++p;
          for (; p < hist.end(be); ++p)
            loop(*p);
        }  // for i
      }
      if (threadIdx.x == 0)
        ++nloops;
    }  // while

    // collect edges (assign to closest cluster of closest point??? here to closest point)
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      //    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
      if (nn[i] >= minT)
        continue;  // DBSCAN edge rule
      float mdist = eps;
      auto loop = [&](int j) {
        if (nn[j] < minT)
          return;  // DBSCAN core rule
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > mdist)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;  // needed?
        mdist = dist;
        iv[i] = iv[j];  // assign to cluster (better be unique??)
      };
      cms::cuda::forEachInBins(hist, izt[i], 1, loop);
    }

    __shared__ unsigned int foundClusters;
    foundClusters = 0;
    __syncthreads();

    // find the number of different clusters, identified by a tracks with clus[i] == i;
    // mark these tracks with a negative id.
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] == int(i)) {
        if (nn[i] >= minT) {
          auto old = atomicInc(&foundClusters, 0xffffffff);
          iv[i] = -(old + 1);
        } else {  // noise
          iv[i] = -9998;
        }
      }
    }
    __syncthreads();

    assert(foundClusters < zVertex::utilities::MAXVTX);

    // propagate the negative id to all the tracks in the cluster.
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] >= 0) {
        // mark each track in a cluster with the same id as the first one
        iv[i] = iv[iv[i]];
      }
    }
    __syncthreads();

    // adjust the cluster id to be a positive value starting from 0
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      iv[i] = -iv[i] - 1;
    }

    nvIntermediate = nvFinal = foundClusters;

    if (verbose && 0 == threadIdx.x)
      printf("found %d proto vertices\n", foundClusters);
  }

}  // namespace gpuVertexFinder

#endif  // RecoTracker_PixelVertexFinding_plugins_gpuClusterTracksIterative_h
