#ifndef RecoPixelVertexing_PixelVertexFinding_alpaka_clusterTracksByDensity_h
#define RecoPixelVertexing_PixelVertexFinding_alpaka_clusterTracksByDensity_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "vertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace vertexFinder {
    using VtxSoAView = ::reco::ZVertexSoAView;
    using WsSoAView = ::vertexFinder::PixelVertexWorkSpaceSoAView;
    // this algo does not really scale as it works in a single block...
    // enough for <10K tracks we have
    //
    // based on Rodrighez&Laio algo
    //
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void __attribute__((always_inline))
    clusterTracksByDensity(const TAcc& acc,
                           VtxSoAView& pdata,
                           WsSoAView& pws,
                           int minT,      // min number of neighbours to be "seed"
                           float eps,     // max absolute distance to cluster
                           float errmax,  // max error to be "seed"
                           float chi2max  // max normalized distance to cluster
    ) {
      using namespace vertexFinder;
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

      if constexpr (verbose) {
        if (cms::alpakatools::once_per_block(acc))
          printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);
      }
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

      ALPAKA_ASSERT_OFFLOAD(zt);
      ALPAKA_ASSERT_OFFLOAD(ezt2);
      ALPAKA_ASSERT_OFFLOAD(izt);
      ALPAKA_ASSERT_OFFLOAD(nn);
      ALPAKA_ASSERT_OFFLOAD(iv);

      using Hist = cms::alpakatools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;
      auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
      auto& hws = alpaka::declareSharedVar<Hist::Counter[32], __COUNTER__>(acc);

      for (auto j : cms::alpakatools::elements_with_stride(acc, Hist::totbins())) {
        hist.off[j] = 0;
      }
      alpaka::syncBlockThreads(acc);

      if constexpr (verbose) {
        if (cms::alpakatools::once_per_block(acc))
          printf("booked hist with %d bins, size %d for %d tracks\n", hist.totbins(), hist.capacity(), nt);
      }
      ALPAKA_ASSERT_OFFLOAD(static_cast<int>(nt) <= hist.capacity());

      // fill hist  (bin shall be wider than "eps")
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        ALPAKA_ASSERT_OFFLOAD(i < ::zVertex::MAXTRACKS);
        int iz = int(zt[i] * 10.);  // valid if eps<=0.1
        // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
        iz = std::min(std::max(iz, INT8_MIN), INT8_MAX);
        izt[i] = iz - INT8_MIN;
        ALPAKA_ASSERT_OFFLOAD(iz - INT8_MIN >= 0);
        ALPAKA_ASSERT_OFFLOAD(iz - INT8_MIN < 256);
        hist.count(acc, izt[i]);
        iv[i] = i;
        nn[i] = 0;
      }
      alpaka::syncBlockThreads(acc);
      if (threadIdxLocal < 32)
        hws[threadIdxLocal] = 0;  // used by prefix scan...
      alpaka::syncBlockThreads(acc);
      hist.finalize(acc, hws);
      alpaka::syncBlockThreads(acc);
      ALPAKA_ASSERT_OFFLOAD(hist.size() == nt);
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        hist.fill(acc, izt[i], uint16_t(i));
      }
      alpaka::syncBlockThreads(acc);
      // count neighbours
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
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

        cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
      }
      alpaka::syncBlockThreads(acc);

      // find closest above me .... (we ignore the possibility of two j at same distance from i)
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        float mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < nn[i])
            return;
          if (nn[j] == nn[i] && zt[j] >= zt[i])
            return;  // if equal use natural order...
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;  // (break natural order???)
          mdist = dist;
          iv[i] = j;  // assign to cluster (better be unique??)
        };
        cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
      }
      alpaka::syncBlockThreads(acc);

#ifdef GPU_DEBUG
      //  mini verification
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] != int(i))
          ALPAKA_ASSERT_OFFLOAD(iv[iv[i]] != int(i));
      }
      alpaka::syncBlockThreads(acc);
#endif

      // consolidate graph (percolate index of seed)
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        auto m = iv[i];
        while (m != iv[m])
          m = iv[m];
        iv[i] = m;
      }

#ifdef GPU_DEBUG
      alpaka::syncBlockThreads(acc);
      //  mini verification
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] != int(i))
          ALPAKA_ASSERT_OFFLOAD(iv[iv[i]] != int(i));
      }
#endif

#ifdef GPU_DEBUG
      // and verify that we did not spit any cluster...
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        auto minJ = i;
        auto mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < nn[i])
            return;
          if (nn[j] == nn[i] && zt[j] >= zt[i])
            return;  // if equal use natural order...
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;
          mdist = dist;
          minJ = j;
        };
        cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
        // should belong to the same cluster...
        ALPAKA_ASSERT_OFFLOAD(iv[i] == iv[minJ]);
        ALPAKA_ASSERT_OFFLOAD(nn[i] <= nn[iv[i]]);
      }
      alpaka::syncBlockThreads(acc);
#endif

      auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
      foundClusters = 0;
      alpaka::syncBlockThreads(acc);

      // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
      // mark these tracks with a negative id.
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] == int(i)) {
          if (nn[i] >= minT) {
            auto old = alpaka::atomicInc(acc, &foundClusters, 0xffffffff, alpaka::hierarchy::Threads{});
            iv[i] = -(old + 1);
          } else {  // noise
            iv[i] = -9998;
          }
        }
      }
      alpaka::syncBlockThreads(acc);

      ALPAKA_ASSERT_OFFLOAD(foundClusters < ::zVertex::MAXVTX);

      // propagate the negative id to all the tracks in the cluster.
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] >= 0) {
          // mark each track in a cluster with the same id as the first one
          iv[i] = iv[iv[i]];
        }
      }
      alpaka::syncBlockThreads(acc);

      // adjust the cluster id to be a positive value starting from 0
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        iv[i] = -iv[i] - 1;
      }

      nvIntermediate = nvFinal = foundClusters;
      if constexpr (verbose) {
        if (cms::alpakatools::once_per_block(acc))
          printf("found %d proto vertices\n", foundClusters);
      }
    }
    class ClusterTracksByDensityKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VtxSoAView pdata,
                                    WsSoAView pws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster
      ) const {
        clusterTracksByDensity(acc, pdata, pws, minT, eps, errmax, chi2max);
      }
    };
  }  // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // RecoPixelVertexing_PixelVertexFinding_alpaka_clusterTracksByDensity_h
