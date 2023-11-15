#ifndef RecoPixelVertexing_PixelVertexFinding_gpuFitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_gpuFitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

#include "vertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace vertexFinder {
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void fitVertices(
        const TAcc& acc,
        VtxSoAView& pdata,
        WsSoAView& pws,
        float chi2Max  // for outlier rejection
    ) {
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false

      auto& __restrict__ data = pdata;
      auto& __restrict__ ws = pws;
      auto nt = ws.ntrks();
      float const* __restrict__ zt = ws.zt();
      float const* __restrict__ ezt2 = ws.ezt2();
      float* __restrict__ zv = data.zv();
      float* __restrict__ wv = data.wv();
      float* __restrict__ chi2 = data.chi2();
      uint32_t& nvFinal = data.nvFinal();
      uint32_t& nvIntermediate = ws.nvIntermediate();

      int32_t* __restrict__ nn = data.ndof();
      int32_t* __restrict__ iv = ws.iv();

      ALPAKA_ASSERT_OFFLOAD(nvFinal <= nvIntermediate);
      nvFinal = nvIntermediate;
      auto foundClusters = nvFinal;

      // zero
      for (auto i : cms::alpakatools::elements_with_stride(acc, foundClusters)) {
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
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] > 9990) {
          if constexpr (verbose)
            alpaka::atomicAdd(acc, &noise, 1, alpaka::hierarchy::Threads{});
          continue;
        }
        ALPAKA_ASSERT_OFFLOAD(iv[i] >= 0);
        ALPAKA_ASSERT_OFFLOAD(iv[i] < int(foundClusters));
        auto w = 1.f / ezt2[i];
        alpaka::atomicAdd(acc, &zv[iv[i]], zt[i] * w, alpaka::hierarchy::Threads{});
        alpaka::atomicAdd(acc, &wv[iv[i]], w, alpaka::hierarchy::Threads{});
      }

      alpaka::syncBlockThreads(acc);
      // reuse nn
      for (auto i : cms::alpakatools::elements_with_stride(acc, foundClusters)) {
        ALPAKA_ASSERT_OFFLOAD(wv[i] > 0.f);
        zv[i] /= wv[i];
        nn[i] = -1;  // ndof
      }
      alpaka::syncBlockThreads(acc);

      // compute chi2
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
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

      for (auto i : cms::alpakatools::elements_with_stride(acc, foundClusters)) {
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
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VtxSoAView pdata,
                                    WsSoAView pws,
                                    float chi2Max  // for outlier rejection
      ) const {
        fitVertices(acc, pdata, pws, chi2Max);
      }
    };
  }  // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // RecoPixelVertexing_PixelVertexFinding_plugins_gpuFitVertices_h
