#ifndef RecoVertex_PixelVertexFinding_plugins_alpaka_clusterTracksByDensity_h
#define RecoVertex_PixelVertexFinding_plugins_alpaka_clusterTracksByDensity_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoVertex/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"

#include "vertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder {

  // This algo does not really scale as it works in a single block...
  // It should be good enough for <10K tracks we have.
  //
  // Based on Rodrighez&Laio algo.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void clusterTracksByDensity(Acc1D const& acc,
                                                             VtxSoAView& data,
                                                             TrkSoAView& trkdata,
                                                             WsSoAView& ws,
                                                             int minT,      // min number of neighbours to be "seed"
                                                             float eps,     // max absolute distance to cluster
                                                             float errmax,  // max error to be "seed"
                                                             float chi2max  // max normalized distance to cluster
  ) {
    constexpr bool verbose = false;

    if constexpr (verbose) {
      if (cms::alpakatools::once_per_block(acc))
        printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);
    }

    auto nt = ws.ntrks();
    ALPAKA_ASSERT_ACC(static_cast<int>(nt) <= ws.metadata().size());
    ALPAKA_ASSERT_ACC(static_cast<int>(nt) <= trkdata.metadata().size());

    float const* __restrict__ zt = ws.zt();
    float const* __restrict__ ezt2 = ws.ezt2();
    uint8_t* __restrict__ izt = ws.izt();
    int32_t* __restrict__ iv = ws.iv();
    int32_t* __restrict__ nn = trkdata.ndof();
    ALPAKA_ASSERT_ACC(zt);
    ALPAKA_ASSERT_ACC(ezt2);
    ALPAKA_ASSERT_ACC(izt);
    ALPAKA_ASSERT_ACC(iv);
    ALPAKA_ASSERT_ACC(nn);

    using Hist = cms::alpakatools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;
    auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
    constexpr int warpSize = cms::alpakatools::warpSize;
    auto& hws = alpaka::declareSharedVar<Hist::Counter[warpSize], __COUNTER__>(acc);

    for (auto j : cms::alpakatools::uniform_elements(acc, Hist::totbins())) {
      hist.off[j] = 0;
    }
    for (auto j : cms::alpakatools::uniform_elements(acc, warpSize)) {
      hws[j] = 0;  // used by prefix scan in hist.finalize()
    }
    alpaka::syncBlockThreads(acc);

    if constexpr (verbose) {
      if (cms::alpakatools::once_per_block(acc))
        printf("booked hist with %d bins, size %d for %d tracks\n", hist.totbins(), hist.capacity(), nt);
    }

    ALPAKA_ASSERT_ACC(static_cast<int>(nt) <= std::numeric_limits<Hist::index_type>::max());
    ALPAKA_ASSERT_ACC(static_cast<int>(nt) <= hist.capacity());
    ALPAKA_ASSERT_ACC(eps <= 0.1f);  // see below

    // fill hist (bin shall be wider than "eps")
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      int iz = static_cast<int>(zt[i] * 10.f);  // valid if eps <= 0.1
      // Equivalent of iz = std::clamp(iz, INT8_MIN, INT8_MAX)
      // which doesn't compile with gcc14 due to reference to __glibcxx_assert
      // See https://github.com/llvm/llvm-project/issues/95183
      int tmp_max = std::max<int>(iz, INT8_MIN);
      iz = std::min<int>(tmp_max, INT8_MAX);
      ALPAKA_ASSERT_ACC(iz - INT8_MIN >= 0);
      ALPAKA_ASSERT_ACC(iz - INT8_MIN < 256);
      izt[i] = iz - INT8_MIN;
      hist.count(acc, izt[i]);
      iv[i] = i;
      nn[i] = 0;
    }
    alpaka::syncBlockThreads(acc);

    hist.finalize(acc, hws);
    alpaka::syncBlockThreads(acc);

    ALPAKA_ASSERT_ACC(hist.size() == nt);
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      hist.fill(acc, izt[i], uint16_t(i));
    }
    alpaka::syncBlockThreads(acc);

    // count neighbours
    const auto errmax2 = errmax * errmax;
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      if (ezt2[i] > errmax2)
        continue;
      cms::alpakatools::forEachInBins(hist, izt[i], 1, [&](uint32_t j) {
        if (i == j)
          return;
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > eps)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;
        nn[i]++;
      });
    }
    alpaka::syncBlockThreads(acc);

    // find closest above me .... (we ignore the possibility of two j at same distance from i)
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      float mdist = eps;
      cms::alpakatools::forEachInBins(hist, izt[i], 1, [&](uint32_t j) {
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
      });
    }
    alpaka::syncBlockThreads(acc);

#ifdef GPU_DEBUG
    //  mini verification
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      if (iv[i] != int(i))
        ALPAKA_ASSERT_ACC(iv[iv[i]] != int(i));
    }
    alpaka::syncBlockThreads(acc);
#endif

    // consolidate graph (percolate index of seed)
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      auto m = iv[i];
      while (m != iv[m])
        m = iv[m];
      iv[i] = m;
    }

#ifdef GPU_DEBUG
    alpaka::syncBlockThreads(acc);
    //  mini verification
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      if (iv[i] != int(i))
        ALPAKA_ASSERT_ACC(iv[iv[i]] != int(i));
    }
#endif

#ifdef GPU_DEBUG
    // and verify that we did not split any cluster...
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      auto minJ = i;
      auto mdist = eps;
      cms::alpakatools::forEachInBins(hist, izt[i], 1, [&](uint32_t j) {
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
      });

      // should belong to the same cluster...
      ALPAKA_ASSERT_ACC(iv[i] == iv[minJ]);
      ALPAKA_ASSERT_ACC(nn[i] <= nn[iv[i]]);
    }
    alpaka::syncBlockThreads(acc);
#endif

    auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
    foundClusters = 0;
    alpaka::syncBlockThreads(acc);

    // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
    // mark these tracks with a negative id.
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
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

    ALPAKA_ASSERT_ACC(static_cast<int>(foundClusters) < data.metadata().size());

    // propagate the negative id to all the tracks in the cluster.
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      if (iv[i] >= 0) {
        // mark each track in a cluster with the same id as the first one
        iv[i] = iv[iv[i]];
      }
    }
    alpaka::syncBlockThreads(acc);

    // adjust the cluster id to be a positive value starting from 0
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      iv[i] = -iv[i] - 1;
    }

    ws.nvIntermediate() = foundClusters;
    data.nvFinal() = foundClusters;

    if constexpr (verbose) {
      if (cms::alpakatools::once_per_block(acc))
        printf("found %d proto vertices\n", foundClusters);
    }
  }

  class ClusterTracksByDensityKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  VtxSoAView data,
                                  TrkSoAView trkdata,
                                  WsSoAView ws,
                                  int minT,      // min number of neighbours to be "seed"
                                  float eps,     // max absolute distance to cluster
                                  float errmax,  // max error to be "seed"
                                  float chi2max  // max normalized distance to cluster
    ) const {
      clusterTracksByDensity(acc, data, trkdata, ws, minT, eps, errmax, chi2max);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder

#endif  // RecoVertex_PixelVertexFinding_plugins_alpaka_clusterTracksByDensity_h
