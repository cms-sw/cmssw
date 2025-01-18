#ifndef RecoVertex_PixelVertexFinding_plugins_alpaka_fitVertices_h
#define RecoVertex_PixelVertexFinding_plugins_alpaka_fitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "vertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder {

  ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void fitVertices(Acc1D const& acc,
                                                                                 VtxSoAView& pdata,
                                                                                 TrkSoAView& ptrkdata,
                                                                                 WsSoAView& pws,
                                                                                 float chi2Max  // for outlier rejection
  ) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    auto& __restrict__ data = pdata;
    auto& __restrict__ trkdata = ptrkdata;
    auto& __restrict__ ws = pws;
    auto nt = ws.ntrks();
    float const* __restrict__ zt = ws.zt();
    float const* __restrict__ ezt2 = ws.ezt2();
    float* __restrict__ zv = data.zv();
    float* __restrict__ wv = data.wv();
    float* __restrict__ chi2 = data.chi2();
    uint32_t& nvFinal = data.nvFinal();
    uint32_t& nvIntermediate = ws.nvIntermediate();

    int32_t* __restrict__ nn = trkdata.ndof();
    int32_t* __restrict__ iv = ws.iv();

    ALPAKA_ASSERT_ACC(nvFinal <= nvIntermediate);
    nvFinal = nvIntermediate;
    auto foundClusters = nvFinal;

    // zero
    for (auto i : cms::alpakatools::uniform_elements(acc, foundClusters)) {
      zv[i] = 0;
      wv[i] = 0;
      chi2[i] = 0;
    }

    // only for test
    auto& noise = alpaka::declareSharedVar<int, __COUNTER__>(acc);

    if constexpr (verbose) {
      if (cms::alpakatools::once_per_block(acc))
        noise = 0;
    }
    alpaka::syncBlockThreads(acc);

    // compute cluster location
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      if (iv[i] > 9990) {
        if constexpr (verbose)
          alpaka::atomicAdd(acc, &noise, 1, alpaka::hierarchy::Threads{});
        continue;
      }
      ALPAKA_ASSERT_ACC(iv[i] >= 0);
      ALPAKA_ASSERT_ACC(iv[i] < int(foundClusters));
      auto w = 1.f / ezt2[i];
      alpaka::atomicAdd(acc, &zv[iv[i]], zt[i] * w, alpaka::hierarchy::Threads{});
      alpaka::atomicAdd(acc, &wv[iv[i]], w, alpaka::hierarchy::Threads{});
    }

    alpaka::syncBlockThreads(acc);
    // reuse nn
    for (auto i : cms::alpakatools::uniform_elements(acc, foundClusters)) {
      if constexpr (verbose) {
        if (not(wv[i] > 0.f)) {
          printf("ERROR: wv[%d] (%f) > 0.f failed\n", i, wv[i]);
          // printing info on tracks associated to this vertex
          for (auto trk_i = 0u; trk_i < nt; ++trk_i) {
            if (iv[trk_i] != int(i)) {
              continue;
            }
            printf("   iv[%d]=%d zt[%d]=%f ezt2[%d]=%f\n", trk_i, iv[trk_i], trk_i, zt[trk_i], trk_i, ezt2[trk_i]);
          }
        }
      }
      ALPAKA_ASSERT_ACC(wv[i] > 0.f);
      zv[i] /= wv[i];
      nn[i] = -1;  // ndof
    }
    alpaka::syncBlockThreads(acc);

    // compute chi2
    for (auto i : cms::alpakatools::uniform_elements(acc, nt)) {
      if (iv[i] > 9990)
        continue;

      auto c2 = zv[iv[i]] - zt[i];
      c2 *= c2 / ezt2[i];
      if (c2 > chi2Max) {
        iv[i] = 9999;
        continue;
      }
      alpaka::atomicAdd(acc, &chi2[iv[i]], c2, alpaka::hierarchy::Blocks{});
      alpaka::atomicAdd(acc, &nn[iv[i]], 1, alpaka::hierarchy::Blocks{});
    }
    alpaka::syncBlockThreads(acc);

    for (auto i : cms::alpakatools::uniform_elements(acc, foundClusters)) {
      if (nn[i] > 0) {
        wv[i] *= float(nn[i]) / chi2[i];
      }
    }
    if constexpr (verbose) {
      if (cms::alpakatools::once_per_block(acc)) {
        printf("found %d proto clusters ", foundClusters);
        printf("and %d noise\n", noise);
      }
    }
  }

  class FitVerticesKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  VtxSoAView pdata,
                                  TrkSoAView ptrkdata,
                                  WsSoAView pws,
                                  float chi2Max  // for outlier rejection
    ) const {
      fitVertices(acc, pdata, ptrkdata, pws, chi2Max);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder

#endif  // RecoVertex_PixelVertexFinding_plugins_alpaka_fitVertices_h
