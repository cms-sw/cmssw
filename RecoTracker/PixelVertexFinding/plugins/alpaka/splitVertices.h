#ifndef RecoPixelVertexing_PixelVertexFinding_splitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_splitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

#include "vertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace vertexFinder {
    using VtxSoAView = ::reco::ZVertexSoAView;
    using WsSoAView = ::vertexFinder::PixelVertexWorkSpaceSoAView;
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void splitVertices(const TAcc& acc,
                                                                                     VtxSoAView& pdata,
                                                                                     WsSoAView& pws,
                                                                                     float maxChi2) {
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

      auto& __restrict__ data = pdata;
      auto& __restrict__ ws = pws;
      auto nt = ws.ntrks();
      float const* __restrict__ zt = ws.zt();
      float const* __restrict__ ezt2 = ws.ezt2();
      float* __restrict__ zv = data.zv();
      float* __restrict__ wv = data.wv();
      float const* __restrict__ chi2 = data.chi2();
      uint32_t& nvFinal = data.nvFinal();

      int32_t const* __restrict__ nn = data.ndof();
      int32_t* __restrict__ iv = ws.iv();

      ALPAKA_ASSERT_OFFLOAD(zt);
      ALPAKA_ASSERT_OFFLOAD(wv);
      ALPAKA_ASSERT_OFFLOAD(chi2);
      ALPAKA_ASSERT_OFFLOAD(nn);

      constexpr uint32_t MAXTK = 512;

      auto& it = alpaka::declareSharedVar<uint32_t[MAXTK], __COUNTER__>(acc);   // track index
      auto& zz = alpaka::declareSharedVar<float[MAXTK], __COUNTER__>(acc);      // z pos
      auto& newV = alpaka::declareSharedVar<uint8_t[MAXTK], __COUNTER__>(acc);  // 0 or 1
      auto& ww = alpaka::declareSharedVar<float[MAXTK], __COUNTER__>(acc);      // z weight
      auto& nq = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);          // number of track for this vertex

      const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

      // one vertex per block
      for (auto kv = blockIdx; kv < nvFinal; kv += gridDimension) {
        if (nn[kv] < 4)
          continue;
        if (chi2[kv] < maxChi2 * float(nn[kv]))
          continue;

        ALPAKA_ASSERT_OFFLOAD(nn[kv] < int32_t(MAXTK));

        if ((uint32_t)nn[kv] >= MAXTK)
          continue;  // too bad FIXME

        nq = 0u;
        alpaka::syncBlockThreads(acc);

        // copy to local
        for (auto k : cms::alpakatools::independent_group_elements(acc, nt)) {
          if (iv[k] == int(kv)) {
            auto old = alpaka::atomicInc(acc, &nq, MAXTK, alpaka::hierarchy::Threads{});
            zz[old] = zt[k] - zv[kv];
            newV[old] = zz[old] < 0 ? 0 : 1;
            ww[old] = 1.f / ezt2[k];
            it[old] = k;
          }
        }

        // the new vertices
        auto& znew = alpaka::declareSharedVar<float[2], __COUNTER__>(acc);
        auto& wnew = alpaka::declareSharedVar<float[2], __COUNTER__>(acc);
        alpaka::syncBlockThreads(acc);

        ALPAKA_ASSERT_OFFLOAD(int(nq) == nn[kv] + 1);

        int maxiter = 20;
        // kt-min....
        bool more = true;
        while (alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, more)) {
          more = false;
          if (0 == threadIdxLocal) {
            znew[0] = 0;
            znew[1] = 0;
            wnew[0] = 0;
            wnew[1] = 0;
          }
          alpaka::syncBlockThreads(acc);

          for (auto k : cms::alpakatools::elements_with_stride(acc, nq)) {
            auto i = newV[k];
            alpaka::atomicAdd(acc, &znew[i], zz[k] * ww[k], alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(acc, &wnew[i], ww[k], alpaka::hierarchy::Threads{});
          }
          alpaka::syncBlockThreads(acc);

          if (0 == threadIdxLocal) {
            znew[0] /= wnew[0];
            znew[1] /= wnew[1];
          }
          alpaka::syncBlockThreads(acc);

          for (auto k : cms::alpakatools::elements_with_stride(acc, nq)) {
            auto d0 = fabs(zz[k] - znew[0]);
            auto d1 = fabs(zz[k] - znew[1]);
            auto newer = d0 < d1 ? 0 : 1;
            more |= newer != newV[k];
            newV[k] = newer;
          }
          --maxiter;
          if (maxiter <= 0)
            more = false;
        }

        // avoid empty vertices
        if (0 == wnew[0] || 0 == wnew[1])
          continue;

        // quality cut
        auto dist2 = (znew[0] - znew[1]) * (znew[0] - znew[1]);

        auto chi2Dist = dist2 / (1.f / wnew[0] + 1.f / wnew[1]);

        if (verbose && 0 == threadIdxLocal)
          printf("inter %d %f %f\n", 20 - maxiter, chi2Dist, dist2 * wv[kv]);

        if (chi2Dist < 4)
          continue;

        // get a new global vertex
        auto& igv = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
        if (0 == threadIdxLocal)
          igv = alpaka::atomicAdd(acc, &ws.nvIntermediate(), 1u, alpaka::hierarchy::Blocks{});
        alpaka::syncBlockThreads(acc);
        for (auto k : cms::alpakatools::elements_with_stride(acc, nq)) {
          if (1 == newV[k])
            iv[it[k]] = igv;
        }

      }  // loop on vertices
    }

    class SplitVerticesKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc, VtxSoAView pdata, WsSoAView pws, float maxChi2) const {
        splitVertices(acc, pdata, pws, maxChi2);
      }
    };
  }  // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // RecoPixelVertexing_PixelVertexFinding_plugins_splitVertices.h
