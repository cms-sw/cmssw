#ifndef RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h

// ============================================================================
// Shared BL-fit device kernels + the per-N launcher helper for the SPLIT build.
//
// Compiling every heavy per-N fit kernel into one nvcc TU is slow: the CA main
// fit instantiates its own N=3..10 kernels beside the fast-fit kernels. This
// header therefore holds every heavy kernel struct plus the per-N launcher
// helper runMainBin<N,T>. It is
// declared `extern template` here so the orchestration TU
// (BrokenLineFit.dev.cc) only EMITS CALLS to it; it is explicitly
// instantiated in exactly one small N-range TU (BrokenLineFit_*.dev.cc), so
// nvcc compiles disjoint kernel subsets in parallel. The split is a build-time
// division only: the kernels and their launch arguments are the same either way.
// ============================================================================

// #define BROKENLINE_DEBUG
// #define GPU_DEBUG
// #define FIT_DEBUG
#include <cstdint>

#if defined(FIT_DEBUG) || defined(GPU_DEBUG)
#include <iostream>
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

#include "CAFitHitSelection.h"
#include "HelixFit.h"

using OutputSoAView = reco::TrackSoAView;
using TupleMultiplicity = caStructures::GenericContainer;
using Tuples = caStructures::SequentialContainer;
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <int N>
  class Kernel_BLFastFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  caStructures::HitsMultiView hh,
                                  ::reco::CAModulesConstView cm,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  uint32_t nHitsL,
                                  uint32_t nHitsH,
                                  int32_t offset,
                                  bool hasStubs) const {
      constexpr uint32_t hitsInFit = N;
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();

      ALPAKA_ASSERT_ACC(hitsInFit <= nHitsL);
      ALPAKA_ASSERT_ACC(nHitsL <= nHitsH);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);

      // look in bin for this hit multiplicity
      int totTK = tupleMultiplicity->end(nHitsH) - tupleMultiplicity->begin(nHitsL);
      ALPAKA_ASSERT_ACC(totTK <= int(tupleMultiplicity->size()));
      ALPAKA_ASSERT_ACC(totTK >= 0);

#ifdef BROKENLINE_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("%d total Ntuple\n", tupleMultiplicity->size());
        printf("%d Ntuple of size %d/%d for %d hits to fit\n", totTK, nHitsL, nHitsH, hitsInFit);
      }
#endif
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if ((int)tuple_idx >= totTK) {
          ptkids[local_idx] = invalidTkId;
          break;
        }
        // get it from the ntuple container (one to one to helix)
        auto tkid = *(tupleMultiplicity->begin(nHitsL) + tuple_idx);
        ALPAKA_ASSERT_ACC(tkid < foundNtuplets->nOnes());

        ptkids[local_idx] = tkid;

        auto nHits = foundNtuplets->size(tkid);

        // multiplicity binning / assertions use the SELECTED hit count (nSel), computed below.
        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        // Prepare data structure
        auto const* hitId = foundNtuplets->begin(tkid);

        // Number of hits the fit uses: kMode filter + same-layer PIXEL overlap dedup (OT stubs never
        // merged). Computed by the SAME caFitHitSel::dedupWalk used in count/fillMultiplicity
        // (CAHitNtupletGeneratorKernelsImpl.h), so the multiplicity bin nSel in [nHitsL,nHitsH] holds.
        // hasStubs is passed in from the launcher as the compile-time OT-stubs topology flag
        // (std::is_same_v<Phase2OTStubs, TrackerTraits>). Combine it with the runtime stub offset the same way
        // nSelectedHits does in the count/fill kernels: a Phase2OTStubs collection carrying no stubs leaves
        // offsetStubs at the unsigned sentinel (-1 as int32), so no OT-stub treatment applies and the bin matches.
        const bool hasStubsRt = hasStubs && (static_cast<int32_t>(hh.view(0).offsetStubs()) >= 0);
        uint32_t nSel = caFitHitSel::dedupWalk(foundNtuplets, tkid, hh, hasStubsRt, /*k=*/-1);
        ALPAKA_ASSERT_ACC(nSel >= nHitsL);
        ALPAKA_ASSERT_ACC(nSel <= nHitsH);

        // Uniform sampling: select hitsInFit hits uniformly from the DEDUPED selected hits.
        uint32_t selectedHits[N];
        uint32_t nSelected = 0;
        {
          float incr = std::max(1.f, float(nSel) / float(hitsInFit));
          float fn = 0;
          for (uint32_t i = 0; i < hitsInFit; ++i) {
            int k = int(fn + 0.5f);  // round -> k-th kept hit
            if (hitsInFit - 1 == i)
              k = int(nSel) - 1;  // force last kept hit (max lever arm)
            // map the k-th kept hit to its position j in [0, nHits) via the shared dedup walk
            uint32_t j = caFitHitSel::dedupWalk(foundNtuplets, tkid, hh, hasStubsRt, k);
            ALPAKA_ASSERT_ACC(j < nHits);
            selectedHits[nSelected++] = j;
            fn += incr;
          }
        }
        ALPAKA_ASSERT_ACC(nSelected == hitsInFit);

        for (uint32_t i = 0; i < hitsInFit; ++i) {
          int j = selectedHits[i];
          auto hit = hitId[j];
          float ge[6];

          // Unified per-hit global-error propagation:
          //   - The "picked" sensor per the one-fit-point-per-stub contract is
          //     stored in CAModulesSoA::innerSensorFrame (== detFrame for pixel
          //     modules, the pixel-side sensor for PS stubs, the physically-inner
          //     sensor for SS stubs).
          //   - For SS stubs the global error is built here from the local error
          //     through that frame, at a runtime cost of one additional
          //     `SOAFrame::toGlobal` per fit hit; OTRecHitsSoA carries no
          //     pre-computed global-error columns.
          auto const& frame = cm.innerSensorFrame(hh[hit].detectorIndex());
          float xerrFit = hh[hit].xerrLocal();
          // Stub transverse-error calibration (fit input only -- track finding is untouched). The stub width cut
          // selects lower-sensor hits whose true resolution is better than the raw single-cluster CPE error the
          // merger stores. The per-hit smoothed-residual U-pull widths give the overestimate directly:
          // ~0.68 (2S barrel), 0.80 (PS barrel), 0.92/0.95 (disks). Scaling the local-x VARIANCE by the squared
          // pull width makes the fit weights match the real resolution. hasStubsRt: only a stubs collection
          // that carries stubs has the stub columns populated -- on every other path this is provably inert.
          if (hasStubsRt && reco::isStub(hh, int32_t(hit))) {
            const bool is2S = hh[hit].yerrLocal() > 0.1f;                                // strip vs macro-pixel
            const bool isBarrelHit = alpaka::math::abs(acc, hh[hit].zGlobal()) < 118.f;  // OT barrel vs TEDD
            const float f = is2S ? (isBarrelHit ? 0.4624f : 0.8464f)                     // sigma 0.68 / 0.92
                                 : (isBarrelHit ? 0.64f : 0.9025f);                      // sigma 0.80 / 0.95
            xerrFit *= f;
          }
          // Kleinwort, arXiv:1201.4320 sec. 2.3: scale the 2S outer-tracker stub's
          // strip-length (degenerate/unmeasured) VARIANCE as seen by the fit ONLY. yVarScale == 1.f leaves
          // yerrFit == yerrLocal(); a large scale (e.g. 100) drives 1/sigma_y^2 -> 0, approaching the exact
          // sec. 2.3 zero-weight on the strip direction. The 2S tag reuses
          // the fit's existing discriminator: a stub whose strip-box variance yerrLocal > 0.1 cm^2 (5cm 2S strips
          // ~2 cm^2 vs PS 1.5mm macro-pixels ~2e-3 cm^2). Fit-input only -- the hit SoA yerrLocal() column is
          // untouched, so CA building / doublet-triplet gates / extension-attach acceptance are unaffected.
          float yerrFit = hh[hit].yerrLocal();
          if (hasStubsRt && yVarScale != 1.f && reco::isStub(hh, int32_t(hit)) && yerrFit > 0.1f)
            yerrFit *= yVarScale;
          frame.toGlobal(xerrFit, 0, yerrFit, ge);

          // Fill position - for stubs: use the global position of the lower hit that is stored in the hit SoA
          hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
          hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
        }
        brokenline::fastFit(acc, hits, fast_fit);
#if 0
      printf("Fast Fit: %f, %f, %f, %f\n", fast_fit(0), fast_fit(1), fast_fit(2), fast_fit(3));
#endif

#ifdef BROKENLINE_DEBUG
        // any NaN value should cause the track to be rejected at a later stage
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(0)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(1)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(2)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(3)));
#endif
      }
    }
  };

  // Stride = the launch's concurrent-fit count (= the fit-buffer lane stride). Defaults to the global
  // riemannFit::stride (the main fit).
  // THE CA MAIN FIT: the factorized Blobel circle+line fit, run by every CA iteration of every topology.
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFit {
  public:
    // Device pointer to the uploaded Geant4 material density grid (blMaterialMap, kSize floats).
    const float* __restrict__ rhoMap_ = nullptr;
    // Fit correctness package (producer parameter useFitCorrections; see the block at the head of
    // BrokenLine.h): one thin scatterer per gap charged at that gap's arrival node with the rigid-node
    // guard, the Karimaki-consistent Fisher basis in the covariance blend, the pion 1/beta Highland form,
    // trapezoid quadrature in the material line integral, and the covariance blend completed to the full
    // 3x3. With it off the fit runs the upstream algebra instead. The producer default is true for the
    // Phase2OTStubs topology and false for the others.
    bool fitCorrections_ = false;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  // Per-lane fit scratch: the prepared-data vectors, the shared band block and
                                  // the helper vectors of the factorized fit, held off the kernel stack frame
                                  // (the frame drives the driver's per-thread local-memory reservation).
                                  double* __restrict__ pscratch) const {
      ALPAKA_ASSERT_ACC(results_view.pt().data());
      ALPAKA_ASSERT_ACC(results_view.eta().data());
      ALPAKA_ASSERT_ACC(results_view.chi2().data());
      ALPAKA_ASSERT_ACC(pfast_fit);

      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();

      // same as above...
      // look in bin for this hit multiplicity
      const auto nt = Stride;  // lane count = this launch's buffer stride (main: global stride)
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        if (invalidTkId == ptkids[local_idx])
          break;
        auto tkid = ptkids[local_idx];

        ALPAKA_ASSERT_ACC(tkid < tupleMultiplicity->capacity());

        riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
        riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
        riemannFit::Map6xNfS<N, Stride> hits_ge(phits_ge + local_idx);

        // Factorized Blobel fit: circle + line. All O(N) state (the prepared-data vectors, the shared band
        // block and the helper vectors) lives in this lane's slice of pscratch.
        static_assert(int(N) <= int(TrackerTraits::maxHitsOnTrackForFullFit),
                      "legacy-fit scratch quota is sized at maxHitsOnTrackForFullFit");
        // [element][lane] layout: element e of lane local_idx is at pscratch[e*Stride + local_idx], so a
        // warp reads consecutive lanes at consecutive addresses. Stride = this launch's lane count.
        brokenline::PreparedBrokenLineDataMap<N, Stride> data(pscratch + local_idx);
        brokenline::LegacyFitWorkspaceMap<N, Stride> fitWs(
            pscratch + local_idx + std::size_t(brokenline::kPreparedDataDoubles<N>) * std::size_t(Stride));
        brokenline::karimaki_circle_fit circle;
        riemannFit::LineFit line;
        brokenline::prepareBrokenLineData(acc, hits, fast_fit, bField, rhoMap_, data, fitWs, fitCorrections_);
        brokenline::lineFit(acc, hits_ge, fast_fit, bField, data, line, fitWs, fitCorrections_);
        brokenline::circleFit(acc, hits, hits_ge, fast_fit, bField, data, circle, fitWs, fitCorrections_);
        reco::copyFromCircle(results_view, circle.par, circle.cov, line.par, line.cov, 1.f / float(bField), tkid);
        results_view[tkid].pt() = float(bField) / float(std::abs(circle.par(2)));
        results_view[tkid].eta() = alpaka::math::asinh(acc, line.par(0));
        results_view[tkid].chi2() = (circle.chi2 + line.chi2) / (2 * N - 5);
        results_view[tkid].ndof() = int8_t(2 * N - 5);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------
  // Per-N launcher helper (extern template): it bundles the fast-fit launch + the fit launch for
  // ONE hit multiplicity N (Kernel_BLFastFit + the factorized fit). The captured launch state travels in
  // a small context POD so the helper is a plain function template that can be `extern template`-declared
  // here and explicitly instantiated in a disjoint-N TU.
  // ---------------------------------------------------------------------------------------------

  // Main-fit launch context: the launchBrokenLineKernels locals and HelixFit members the launches need.
  template <typename TrackerTraits>
  struct BLMainLaunchCtx {
    Queue& queue;
    Tuples const* tuples;
    TupleMultiplicity const* tupleMultiplicity;
    caStructures::HitsMultiView hv;
    ::reco::CAModulesConstView cm;
    OutputSoAView outputSoa;
    double bField;
    const float* rhoMap;
    typename caStructures::tindex_type* tkids;
    double* phits;
    float* phits_ge;
    double* pfast_fit;
    double* pgblScratch;
    float yVarScale;      // 2S strip-length variance scale forwarded to Kernel_BLFastFit (1.f = as measured)
    bool fitCorrections;  // fit correctness package (see Kernel_BLFit::fitCorrections_)
  };

  // One (fast-fit + fit) launch of multiplicity N over the current chunk. `wd` is the work division
  // for BOTH the fast-fit and the fit (triplets: workDivTriplets; quads/penta: workDivQuadsPenta).
  // nHitsL/nHitsH are the multiplicity bin of the fast-fit (== N for the exact bins,
  // [maxHitsOnTrackForFullFit, maxHitsOnTrack-1] for the "rest" bin).
  // The CA main fit is the factorized fast BrokenLine fit for every topology: one launch, one
  // linearization.
  template <int N, typename TrackerTraits>
  void runMainBin(
      BLMainLaunchCtx<TrackerTraits> const& c, WorkDiv1D const& wd, uint32_t nHitsL, uint32_t nHitsH, int32_t offset) {
    alpaka::exec<Acc1D>(c.queue,
                        wd,
                        Kernel_BLFastFit<N>{},
                        c.tuples,
                        c.tupleMultiplicity,
                        c.hv,
                        c.cm,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        nHitsL,
                        nHitsH,
                        offset,
                        std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>);
    alpaka::exec<Acc1D>(c.queue,
                        wd,
                        Kernel_BLFit<N, TrackerTraits>{c.rhoMap, c.fitCorrections},
                        c.tupleMultiplicity,
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pgblScratch);
  }

  // Explicit-instantiation signatures (expanded with `extern` below to suppress instantiation in the
  // orchestration TU, and bare in each BrokenLineFit_*.dev.cc to instantiate its disjoint N-subset).
  // The factorized fast BrokenLine main fit, the only fit the CA runs. The stubs N-range (3..10) is
  // compiled in BrokenLineFit_stubsMainBL.dev.cc; the four other traits (3..6) share
  // BrokenLineFit_main.dev.cc.
#define BLFIT_MAIN_SIG_BL(N, T)                    \
  template void runMainBin<N, ::pixelTopology::T>( \
      BLMainLaunchCtx<::pixelTopology::T> const&, WorkDiv1D const&, uint32_t, uint32_t, int32_t)

  // Main fit: N = 3 .. maxHitsOnTrackForFullFit(traits). Non-stubs traits fit up to 6, Phase2OTStubs up to 10.
  extern BLFIT_MAIN_SIG_BL(3, Phase1);
  extern BLFIT_MAIN_SIG_BL(4, Phase1);
  extern BLFIT_MAIN_SIG_BL(5, Phase1);
  extern BLFIT_MAIN_SIG_BL(6, Phase1);
  extern BLFIT_MAIN_SIG_BL(3, Phase2);
  extern BLFIT_MAIN_SIG_BL(4, Phase2);
  extern BLFIT_MAIN_SIG_BL(5, Phase2);
  extern BLFIT_MAIN_SIG_BL(6, Phase2);
  extern BLFIT_MAIN_SIG_BL(3, Phase2OT);
  extern BLFIT_MAIN_SIG_BL(4, Phase2OT);
  extern BLFIT_MAIN_SIG_BL(5, Phase2OT);
  extern BLFIT_MAIN_SIG_BL(6, Phase2OT);
  extern BLFIT_MAIN_SIG_BL(3, HIonPhase1);
  extern BLFIT_MAIN_SIG_BL(4, HIonPhase1);
  extern BLFIT_MAIN_SIG_BL(5, HIonPhase1);
  extern BLFIT_MAIN_SIG_BL(6, HIonPhase1);

  // Phase2OTStubs, N = 3 .. 10.
  extern BLFIT_MAIN_SIG_BL(3, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(4, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(5, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(6, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(7, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(8, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(9, Phase2OTStubs);
  extern BLFIT_MAIN_SIG_BL(10, Phase2OTStubs);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
