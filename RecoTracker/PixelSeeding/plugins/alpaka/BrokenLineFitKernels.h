#ifndef RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h

// Shared BL-fit device kernels + per-N launcher helpers for the SPLIT build.
// Every heavy per-N kernel is `extern template` here and explicitly instantiated in exactly one small
// N-range TU (BrokenLineFit_*.dev.cc), so nvcc compiles disjoint kernel subsets in parallel. Build-time
// division only: the kernels and their launch arguments are unchanged either way.

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
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // unfactorized GBL fit
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"  // (Bz,Br) r-z map for the effective bending field

#include "CAFitHitSelection.h"
#include "HelixFit.h"

using OutputSoAView = reco::TrackSoAView;
using TupleMultiplicity = caStructures::GenericContainer;
using Tuples = caStructures::SequentialContainer;
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // CUDA block dimension of the fit work divisions. Launch dimension only: the fit kernels are
  // grid-stride loops that break on the invalid-tkid sentinel, so no fitted lane and no fitted value
  // depends on it. 32 is the L1-pressure optimum of this solver. The fused main-fit ladder below
  // expresses its bin <-> lane partition in units of it.
  inline constexpr uint32_t kFitBlock = 32u;

  // ---------------------------------------------------------------------------------------------
  // Fused main-fit ladder: the dynamic bin <-> lane-range partition of the CA fit.
  //
  // The main fit's N-bins recycle ONE lane-strided buffer set (tkids / hits / hits_ge / fast_fit /
  // fit scratch, all riemannFit::maxNumberOfConcurrentFits lanes wide), and that -- not any data
  // dependence -- is what forces them to run one after another: bin b+1's fast fit overwrites the
  // lanes bin b's fit is about to read. The bins are disjoint IN TUPLES (tupleMultiplicity bins every
  // tuple by its selected multiplicity, and begin(nHitsL)..end(nHitsH) are contiguous non-overlapping
  // slices of it), so two bins can never write the same SoA row. Here the bins partition the SAME
  // buffer into disjoint lane RANGES instead, and all of them survive to one fused launch per phase.
  //
  // THE PARTITION IS COMPUTED PER EVENT, ON THE DEVICE, FROM THE ACTUAL POPULATIONS -- there is no
  // fixed per-bin quota and nothing crosses to the host: no readback, no sync. No count kernel is
  // needed: tupleMultiplicity is the count, bin b's demand being
  // end(nHitsH) - begin(nHitsL), the very expression the per-bin fast fit's totTK reads. One
  // single-thread kernel (Kernel_BLMainLaneRanges) prefix-sums those demands into contiguous ranges
  // [base_b, base_b + n_b) and writes the block -> bin dispatch table the fused kernels read.
  //
  // WHY THE LAYOUTS SURVIVE UNTOUCHED (this is what makes the partition memory-neutral): every
  // main-fit per-lane buffer is COLUMN-major with the fixed inter-element stride riemannFit::stride,
  // so lane l occupies exactly the elements {l + e*stride} and two lanes cannot share a byte whatever
  // N each of them carries -- hits (Map3xNdS<N, stride>), hits_ge (Map6xNfS<N, stride>), fast_fit
  // (Map4dS<stride>) and the fit scratch (PreparedBrokenLineDataMap<N, stride> +
  // LegacyFitWorkspaceMap<N, stride>) alike. The lane buffers therefore keep the size, the stride and
  // the intra-lane layout they always had, so no per-lane-quota monotonicity argument is needed:
  // disjointness here is unconditional.
  //
  // CAPACITY AND ROUNDS. One pass seats riemannFit::maxNumberOfConcurrentFits lanes. When the demand
  // exceeds that, Kernel_BLMainLaneRanges hands the capacity out in bin order and the bins that did
  // not fit keep their unseated tuples for the next round. A PER-BIN CURSOR makes that exact: the
  // bin -> lane map is deterministic by index here (lane l of bin b is that bin's (tupleBase_b + l -
  // base_b)-th tuple), not an atomic claim, so a per-bin seated count replaces a per-track served
  // flag -- kMainNBins words instead of one byte per track. The round
  // bound is the chunk loop's own bound, unchanged: ceil(chunkBound / maxNumberOfConcurrentFits),
  // and the demand can never exceed the population chunkBound counts, so the rounds always drain it.
  // NO TUPLE IS EVER LEFT UNFIT.
  inline constexpr int kMainMinN = 3;
  template <typename TrackerTraits>
  inline constexpr uint32_t kMainNBins = uint32_t(TrackerTraits::maxHitsOnTrackForFullFit) - uint32_t(kMainMinN) + 1u;

  // Device tables of one round. pRange holds {firstLane, nLanes, tupleBase} per bin; pBlockMap holds
  // {bin, firstLane, endLane} per fused-grid block, with bin == kMainNBins marking an idle block.
  inline constexpr uint32_t kMainRangeStride = 3u;
  inline constexpr uint32_t kMainBlockMapStride = 3u;

  // Fused-grid width. A bin holding n_b lanes needs ceil(n_b / kFitBlock) blocks, and
  //   sum_b ceil(n_b / kFitBlock) <= (sum_b n_b) / kFitBlock + kMainNBins
  //                               <= maxNumberOfConcurrentFits / kFitBlock + kMainNBins,
  // so this width covers EVERY partition the populations can produce. The grid is fixed and
  // host-known; which of its blocks are live, and on which bin, is what the device table decides. A
  // block carries exactly one bin, so the switch over compile-time N never diverges inside a warp.
  template <typename TrackerTraits>
  inline constexpr uint32_t kMainFusedBlocks =
      riemannFit::maxNumberOfConcurrentFits / kFitBlock + kMainNBins<TrackerTraits>;

  // THE ONE DEFINITION of the main ladder's binning, so the range kernel and the fused switch cannot
  // drift apart: it reproduces exactly the (nHitsL, nHitsH) pairs launchBrokenLineKernels launches the
  // per-bin ladder with -- the exact bins N = kMainMinN .. maxHitsOnTrackForFullFit - 1 (rolling_fits
  // is End-EXCLUSIVE), and the top bin N = maxHitsOnTrackForFullFit absorbing the tail
  // [maxHitsOnTrackForFullFit, maxHitsOnTrack - 1].
  template <typename TrackerTraits>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t mainBinNHitsL(uint32_t b) {
    return uint32_t(kMainMinN) + b;
  }
  template <typename TrackerTraits>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t mainBinNHitsH(uint32_t b) {
    return (b + 1u == kMainNBins<TrackerTraits>) ? (uint32_t(TrackerTraits::maxHitsOnTrack) - 1u)
                                                 : (uint32_t(kMainMinN) + b);
  }

  // Out-of-line boundary of the per-lane fit bodies. RecoTracker/PixelSeeding is built with -Ofast
  // (-ffast-math), so the compiler may reassociate FP and form FMAs per function: two inlined copies of
  // the same source (the per-bin ladder's kernel vs the fused ladder's switch) would produce different
  // doubles, and a bordered-band solve amplifies that into a different chi2. Pinned out of line, both
  // ladders call one compiled function. `noclone` keeps -fipa-cp-clone from specialising a copy on one
  // call site's constants.
#if defined(__CUDACC__)
#define BL_REFIT_NOINLINE __noinline__
#elif defined(__clang__)
#define BL_REFIT_NOINLINE __attribute__((noinline))
#elif defined(__GNUC__)
#define BL_REFIT_NOINLINE __attribute__((noinline, noclone))
#else
#define BL_REFIT_NOINLINE
#endif

  template <int N>
  class Kernel_BLFastFit {
  public:
    // Out-of-line per-lane body of the fast fit (BL_REFIT_NOINLINE, see above): the per-bin ladder's
    // operator() and the fused ladder's switch call this one compiled function. No cross-lane or
    // cross-block state: the lane index only forms addresses.
    // Templated on the hit view (HitsMultiView, or the CAHitsView facade of Phase2OTStubs).
    template <typename HitsView>
    ALPAKA_FN_ACC BL_REFIT_NOINLINE static void lane(Acc1D const& acc,
                                                     uint32_t local_idx,
                                                     uint32_t tuple_idx,
                                                     Tuples const* __restrict__ foundNtuplets,
                                                     TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                                     HitsView const& hh,
                                                     ::reco::CAModulesConstView const& cm,
                                                     typename caStructures::tindex_type* __restrict__ ptkids,
                                                     double* __restrict__ phits,
                                                     float* __restrict__ phits_ge,
                                                     double* __restrict__ pfast_fit,
                                                     uint32_t nHitsL,
                                                     uint32_t nHitsH,
                                                     bool hasStubs) {
      constexpr uint32_t hitsInFit = N;
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

      // Number of hits the fit uses: kMode filter + same-layer PIXEL overlap dedup (OT stubs never merged),
      // by the SAME caFitHitSel::dedupWalk used in count/fillMultiplicity. hasStubs is the compile-time
      // OT-stubs topology flag; combine it with the runtime stub offset as count/fill do.
      const bool hasStubsRt = hasStubs && (static_cast<int32_t>(caStructures::offsetStubsOf(hh)) >= 0);
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

        // Unified per-hit global-error propagation. The "picked" sensor per the
        // one-fit-point-per-stub contract is stored in CAModulesSoA::innerSensorFrame (the pixel-side
        // sensor for PS stubs, the physically-inner sensor for SS stubs, == detFrame for pixels). For SS
        // stubs the global error is built here from the local error through that frame (OTRecHitsSoA has no
        // pre-computed global-error columns).
        auto frame = cm.innerSensorFrame(hh[hit].detectorIndex());
        float xerrFit = hh[hit].xerrLocal();
        // Stub transverse-error calibration (fit input only): scale the local-x VARIANCE by the squared
        // pull width (2S barrel 0.68, PS barrel 0.80, disks 0.92/0.95) so the fit weights match the real
        // resolution. Fit-input only -- track finding is untouched.
        if (hasStubsRt && isStub(hh, int32_t(hit))) {
          const bool is2S = hh[hit].yerrLocal() > 0.1f;                                // strip vs macro-pixel
          const bool isBarrelHit = alpaka::math::abs(acc, hh[hit].zGlobal()) < 118.f;  // OT barrel vs TEDD
          const float f = is2S ? (isBarrelHit ? 0.4624f : 0.8464f)                     // sigma 0.68 / 0.92
                               : (isBarrelHit ? 0.64f : 0.9025f);                      // sigma 0.80 / 0.95
          xerrFit *= f;
        }
        // The strip-length variance (degenerate/unmeasured for a 2S stub) enters the fit as measured.
        float yerrFit = hh[hit].yerrLocal();
        frame.toGlobal(xerrFit, 0, yerrFit, ge);

        // Fill position - for stubs: use the global position of the lower hit that is stored in the hit SoA
        hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
        hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
      }
      brokenline::fastFit(acc, hits, fast_fit);

#ifdef BROKENLINE_DEBUG
      // any NaN value should cause the track to be rejected at a later stage
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(0)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(1)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(2)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(3)));
#endif
    }

    // Templated on the hit view, like lane() above.
    template <typename HitsView>
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  HitsView hh,
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
        Kernel_BLFastFit<N>::lane(acc,
                                  local_idx,
                                  uint32_t(tuple_idx),
                                  foundNtuplets,
                                  tupleMultiplicity,
                                  hh,
                                  cm,
                                  ptkids,
                                  phits,
                                  phits_ge,
                                  pfast_fit,
                                  nHitsL,
                                  nHitsH,
                                  hasStubs);
      }
    }
  };

  // Per-track effective field for the curvature->pT conversion. The fit solves a geometric curvature of
  // the transverse projection, whose bending field is B_bend = Bz - Br*tanLambda*cos(alpha) (see
  // BLBFieldMap.h). The hit-average of B_bend/Bz(0,0) scales the origin scalar, with tanLambda*cos(alpha)
  // = (cx*y - cy*x)/(|R|*r) carrying q^2 = 1, so the correction is charge-free. Null bMap -> origin field.
  template <typename TAcc, typename M3xN, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double blEffectiveBField(
      const TAcc& acc, const M3xN& hits, int n, const V4& fast_fit, double bField, const float* bMap) {
    if (bMap == nullptr)
      return bField;
    const double cx = fast_fit(0);
    const double cy = fast_fit(1);
    const double absR = alpaka::math::abs(acc, fast_fit(2));
    const double slopeDen = fast_fit(3) * absR;
    double sSum = 0.;
    for (int i = 0; i < n; ++i) {
      const double x = hits(0, i);
      const double y = hits(1, i);
      const double r = alpaka::math::sqrt(acc, x * x + y * y);
      const double den = slopeDen * r;
      const double tanLambdaCosAlpha = (den != 0.) ? -(cx * y - cy * x) / den : 0.;
      sSum += blBFieldMap::bBendAt(bMap, r, hits(2, i), tanLambdaCosAlpha);
    }
    return bField * (sSum / double(n));
  }

  // Stride = the launch's concurrent-fit count (= the fit-buffer lane stride). Defaults to the global
  // riemannFit::stride (the main fit); a launch with fewer fits instantiates it at its own stride so its
  // buffers stop allocating at the 8192-wide global stride.
  // The CA main fit: the factorized Blobel circle+line fit, run by every CA iteration of every topology.
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFit {
  public:
    // Out-of-line per-lane body of the fit (BL_REFIT_NOINLINE): the per-bin ladder's operator() below and
    // the fused ladder's switch (Kernel_BLFitFused, through blMainFitLaneOutOfLine) call this one compiled
    // function. It fits lane `local_idx` of the shared stride-wide buffers and writes the one SoA row
    // ptkids[local_idx] names.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                              double bField,
                                              OutputSoAView& results_view,
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double* __restrict__ phits,
                                              float* __restrict__ phits_ge,
                                              double* __restrict__ pfast_fit,
                                              double* __restrict__ pscratch) const {
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
      // Per-track effective bending field (blEffectiveBField): the hit-average
      // of B_bend(r,z)/Bz(0,0), charge-free, returning bField exactly where the map is flat. It carries
      // the |z| falloff of Bz and the endcap radial component, i.e. the forward pT bias of a scalar field.
      // Evaluated before the fit state exists so its scalars are dead by prepareBrokenLineData's O(N) set.
      // fitCorrections_ off, or null bMap_, => scalar bField.
      const double bFieldEff = fitCorrections_ ? blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_) : bField;
      // bFieldEff enters as the momentum p = bFieldEff * radius * sqrt(1+slope^2) the Highland variance is
      // divided by, as the 1/bFieldEff of copyFromCircle's geometric-curvature -> q/pT conversion, and as
      // pt = bFieldEff/|curvature|. The two fits additionally take the map itself and the field it is
      // normalized to (the origin scalar bField): the B_r dip-angle row of the line fit and the bending
      // profile's departure from bFieldEff in the circle fit are deterministic offsets of their coordinates,
      // so the single effective scalar keeps its meaning and no fit parameter is added.
      brokenline::prepareBrokenLineData(acc,
                                        hits,
                                        fast_fit,
                                        bFieldEff,
                                        rhoMap_,
                                        data,
                                        fitWs,
                                        fitCorrections_,
                                        /*elossGaps=*/fitCorrections_,
                                        bMap_,
                                        bField);
      brokenline::lineFit(acc, hits_ge, fast_fit, bFieldEff, data, line, fitWs, fitCorrections_);
      // Ionization energy loss: the per-node Landau law of the walked columns, which prepareBrokenLineData
      // left in the workspace under elossGaps; circleFit turns it into the deterministic residual offset,
      // together with the bending profile's departure from bFieldEff.
      brokenline::circleFit(acc,
                            hits,
                            hits_ge,
                            fast_fit,
                            bFieldEff,
                            data,
                            circle,
                            fitWs,
                            fitCorrections_,
                            /*elossGaps=*/fitCorrections_,
                            bMap_,
                            bField);
      // The fit's reference is the pre-fit circle and its abscissa starts at that circle's perigee, while
      // the state is published at the perigee of the trajectory: transport it there, by the arc the fit
      // moved the perigee along and by the field's departure over the lever inside hit 0.
      if (fitCorrections_)
        brokenline::transportToFittedPca(acc, hits, data, bFieldEff, circle, line, bMap_, bField);
      reco::copyFromCircle(results_view, circle.par, circle.cov, line.par, line.cov, 1.f / float(bFieldEff), tkid);
      results_view[tkid].pt() = float(bFieldEff) / float(std::abs(circle.par(2)));
      results_view[tkid].eta() = alpaka::math::asinh(acc, line.par(0));
      results_view[tkid].chi2() = (circle.chi2 + line.chi2) / (2 * N - 5);
      results_view[tkid].ndof() = int8_t(2 * N - 5);
    }

    // Device pointer to the uploaded Geant4 material map (blMaterialMap::Map: density and dE/dx lattices).
    const blMaterialMap::Map* __restrict__ rhoMap_ = nullptr;
    // Device pointer to the (Bz,Br) r-z field map (blBFieldMap), an EventSetup condition. Consumed only
    // under fitCorrections_: the material/field model is one, so the
    // scattering variance is only correct if its momentum comes from the same field that bent the track.
    // Off, or null, => scalar bField (the upstream algebra).
    const float* __restrict__ bMap_ = nullptr;
    // Fit correctness package (producer parameter useFitCorrections; see the head of BrokenLine.h): thin
    // scatterer per gap, rigid-node guard, Karimaki-Fisher covariance blend, pion 1/beta Highland form,
    // trapezoid material quadrature, full 3x3 blend, per-track effective bending field, and ionization
    // energy-loss offset. Off => upstream algebra. Default true for Phase2OTStubs, false otherwise.
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
        lane(acc, local_idx, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------
  // THE DYNAMIC PARTITION of the main fit ladder. Single-threaded, device-only (no host crossing): hands
  // lanes out in bin order, writes the block->bin dispatch table (one bin per block, idle tail at
  // kMainNBins). Nothing is memset -- the kernel writes every word it reads.
  // ---------------------------------------------------------------------------------------------
  template <typename TrackerTraits>
  class Kernel_BLMainLaneRanges {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  uint32_t* __restrict__ pCursor,
                                  uint32_t* __restrict__ pRange,
                                  uint32_t* __restrict__ pBlockMap,
                                  uint32_t laneTotal,
                                  bool firstRound) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      uint32_t base = 0u;
      for (uint32_t b = 0; b < kNBins; ++b) {
        // The bin's population, off the SAME container slice the per-bin fast fit's totTK reads.
        const int32_t tot = int32_t(tupleMultiplicity->end(mainBinNHitsH<TrackerTraits>(b)) -
                                    tupleMultiplicity->begin(mainBinNHitsL<TrackerTraits>(b)));
        const uint32_t have = firstRound ? 0u : pCursor[b];  // seated in earlier rounds
        const uint32_t want = (tot > 0 && uint32_t(tot) > have) ? (uint32_t(tot) - have) : 0u;
        const uint32_t room = (base < laneTotal) ? (laneTotal - base) : 0u;
        const uint32_t got = (want < room) ? want : room;
        pRange[kMainRangeStride * b + 0] = base;
        pRange[kMainRangeStride * b + 1] = got;
        pRange[kMainRangeStride * b + 2] = have;  // first tuple of the bin this round seats
        pCursor[b] = have + got;
        base += got;
      }
      uint32_t blk = 0u;
      for (uint32_t b = 0; b < kNBins; ++b) {
        const uint32_t lo = pRange[kMainRangeStride * b + 0];
        const uint32_t hi = lo + pRange[kMainRangeStride * b + 1];
        for (uint32_t l = lo; l < hi; l += kFitBlock) {
          ALPAKA_ASSERT_ACC(blk < kMainFusedBlocks<TrackerTraits>);  // sum_b ceil(n_b/kFitBlock) <= width
          const uint32_t e = (l + kFitBlock < hi) ? (l + kFitBlock) : hi;
          pBlockMap[kMainBlockMapStride * blk + 0] = b;
          pBlockMap[kMainBlockMapStride * blk + 1] = l;
          pBlockMap[kMainBlockMapStride * blk + 2] = e;
          ++blk;
        }
      }
      for (; blk < kMainFusedBlocks<TrackerTraits>; ++blk) {
        pBlockMap[kMainBlockMapStride * blk + 0] = kNBins;  // idle
        pBlockMap[kMainBlockMapStride * blk + 1] = 0u;
        pBlockMap[kMainBlockMapStride * blk + 2] = 0u;
      }
    }
  };

  // N-independent config of one fused main-fit launch: the per-bin kernels' members + N-independent args.
  // One object serves every bin; the trampoline rebuilds the bin's kernel inside the callee's frame.
  struct BLMainFusedCfg {
    const blMaterialMap::Map* __restrict__ rhoMap = nullptr;
    const float* __restrict__ bMap = nullptr;
    bool fitCorrections = false;
    bool hasStubs = false;
  };

  // The fit's out-of-line trampoline: rebuilds the bin's kernel object from the shared config inside the
  // callee's frame and calls the pinned lane body. (The fast fit needs no trampoline -- Kernel_BLFastFit<N>
  // carries no members, so the fused switch calls its static `lane` directly.)
  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blMainFitLaneOutOfLine(
      Acc1D const& acc,
      BLMainFusedCfg const& cfg,
      uint32_t lane,
      TupleMultiplicity const* __restrict__ tupleMultiplicity,
      double bField,
      OutputSoAView& results_view,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double* __restrict__ phits,
      float* __restrict__ phits_ge,
      double* __restrict__ pfast_fit,
      double* __restrict__ pscratch) {
    const Kernel_BLFit<N, TrackerTraits, Stride> k{cfg.rhoMap, cfg.bMap, cfg.fitCorrections};
    k.lane(acc, lane, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch);
  }

  // The fused main-fit kernels. The block index picks the bin (and its compile-time N) out of the device
  // table Kernel_BLMainLaneRanges wrote for this round; the elements of that block are the bin's lanes.
  // One block carries one bin, so the switch over N cannot diverge inside a warp; an unused block returns
  // at once. Bin b runs the same out-of-line Kernel_BLFastFit/BLFit<b+kMainMinN>::lane the per-bin ladder
  // calls; only the lane a tuple lands in differs, and no lane body reads its lane index beyond its own
  // addresses, so the arithmetic and the set of fitted tuples are the same on both ladders.
  template <typename TrackerTraits>
  struct Kernel_BLFastFitFused {
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    // The switch dispatches bins 0..7 by hand; kMainNBins is traits-derived. A traits set carrying more
    // bins would pass the idle test, fall to `default:` and lose its tuples silently, so raising
    // maxHitsOnTrackForFullFit past 10 must be a build error.
    static_assert(kMainNBins<TrackerTraits> <= 8u,
                  "the fused main ladder dispatches bins 0..7; add cases when a traits set carries more");
    // No Stride template here (unlike Kernel_BLFitFused): the fast fit uses the default-stride maps
    // (riemannFit::stride == maxNumberOfConcurrentFits), the same lane space the fit kernel asserts on.
    BLMainFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  HitsMultiView hh,
                                  ::reco::CAModulesConstView cm,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  const uint32_t* __restrict__ pRange,
                                  const uint32_t* __restrict__ pBlockMap) const {
      // The per-bin kernel's entry assertions (see Kernel_BLFastFit::operator()).
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      constexpr uint32_t kTotal = kMainFusedBlocks<TrackerTraits> * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kMainBlockMapStride * blk];
        if (bin >= kNBins)
          continue;  // idle block: this round's partition did not need it
        ALPAKA_ASSERT_ACC(mainBinNHitsL<TrackerTraits>(bin) <= mainBinNHitsH<TrackerTraits>(bin));
        const uint32_t laneLo = pBlockMap[kMainBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kMainBlockMapStride * blk + 2];
        // Where this bin starts in the lane space, and which of its tuples this round seats first.
        const uint32_t binBase = pRange[kMainRangeStride * bin];
        const uint32_t tupleBase = pRange[kMainRangeStride * bin + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          const uint32_t tuple_idx = tupleBase + (local_idx - binBase);
#define BL_MAIN_FUSED_FAST_CASE(BIN)                                               \
  case (BIN):                                                                      \
    if constexpr (uint32_t(BIN) < kMainNBins<TrackerTraits>) {                     \
      Kernel_BLFastFit<(BIN) + kMainMinN>::lane(acc,                               \
                                                local_idx,                         \
                                                tuple_idx,                         \
                                                foundNtuplets,                     \
                                                tupleMultiplicity,                 \
                                                hh,                                \
                                                cm,                                \
                                                ptkids,                            \
                                                phits,                             \
                                                phits_ge,                          \
                                                pfast_fit,                         \
                                                mainBinNHitsL<TrackerTraits>(BIN), \
                                                mainBinNHitsH<TrackerTraits>(BIN), \
                                                cfg_.hasStubs);                    \
    }                                                                              \
    break
          switch (bin) {
            BL_MAIN_FUSED_FAST_CASE(0);
            BL_MAIN_FUSED_FAST_CASE(1);
            BL_MAIN_FUSED_FAST_CASE(2);
            BL_MAIN_FUSED_FAST_CASE(3);
            BL_MAIN_FUSED_FAST_CASE(4);
            BL_MAIN_FUSED_FAST_CASE(5);
            BL_MAIN_FUSED_FAST_CASE(6);
            BL_MAIN_FUSED_FAST_CASE(7);
            default:
              break;
          }
#undef BL_MAIN_FUSED_FAST_CASE
        }
      }
    }
  };

  template <typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitFused {
    static_assert(Stride == riemannFit::maxNumberOfConcurrentFits,
                  "the fused main ladder spans the main fit's lane buffers");
    // Same guard as Kernel_BLFastFitFused: the switch dispatches bins 0..7 by hand.
    static_assert(kMainNBins<TrackerTraits> <= 8u,
                  "the fused main ladder dispatches bins 0..7; add cases when a traits set carries more");
    BLMainFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  double* __restrict__ pscratch,
                                  const uint32_t* __restrict__ pBlockMap) const {
      ALPAKA_ASSERT_ACC(results_view.pt().data());
      ALPAKA_ASSERT_ACC(results_view.eta().data());
      ALPAKA_ASSERT_ACC(results_view.chi2().data());
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      constexpr uint32_t kTotal = kMainFusedBlocks<TrackerTraits> * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kMainBlockMapStride * blk];
        if (bin >= kNBins)
          continue;  // idle block
        const uint32_t laneLo = pBlockMap[kMainBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kMainBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          // Safety net only: a bin's range IS its exact count and the fused fast fit above filled
          // every lane of it, so the sentinel cannot be reached unless the range and the fill ever
          // disagree.
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_MAIN_FUSED_FIT_CASE(BIN)                                                                                     \
  case (BIN):                                                                                                           \
    if constexpr (uint32_t(BIN) < kMainNBins<TrackerTraits>) {                                                          \
      blMainFitLaneOutOfLine<(BIN) + kMainMinN, TrackerTraits, Stride>(                                                 \
          acc, cfg_, local_idx, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch); \
    }                                                                                                                   \
    break
          switch (bin) {
            BL_MAIN_FUSED_FIT_CASE(0);
            BL_MAIN_FUSED_FIT_CASE(1);
            BL_MAIN_FUSED_FIT_CASE(2);
            BL_MAIN_FUSED_FIT_CASE(3);
            BL_MAIN_FUSED_FIT_CASE(4);
            BL_MAIN_FUSED_FIT_CASE(5);
            BL_MAIN_FUSED_FIT_CASE(6);
            BL_MAIN_FUSED_FIT_CASE(7);
            default:
              break;
          }
#undef BL_MAIN_FUSED_FIT_CASE
        }
      }
    }
  };

  // ---------------------------------------------------------------------------------------------
  // Per-N launcher helpers (extern template). Each bundles the fast-fit + fit launches for ONE
  // multiplicity N. The launch state travels in a small context POD, so the helper is a plain function
  // template that can be `extern template`-declared here and instantiated in a disjoint-N TU.
  // ---------------------------------------------------------------------------------------------

  // Main-fit launch context: the launchBrokenLineKernels locals and HelixFit members the launches need.
  template <typename TrackerTraits>
  struct BLMainLaunchCtx {
    Queue& queue;
    Tuples const* tuples;
    TupleMultiplicity const* tupleMultiplicity;
    caStructures::HitsViewT<TrackerTraits> hv;
    ::reco::CAModulesConstView cm;
    OutputSoAView outputSoa;
    double bField;
    const blMaterialMap::Map* rhoMap;
    const float* bMap;  // normalized (Bz,Br) r-z field map; null (or fitCorrections off) => the scalar bField
    typename caStructures::tindex_type* tkids;
    double* phits;
    float* phits_ge;
    double* pfast_fit;
    double* pgblScratch;
    bool fitCorrections;  // fit correctness package (see Kernel_BLFit::fitCorrections_)
    // Dynamic-partition state of the fused ladder (unused on the per-bin path, which addresses the whole
    // buffer from lane 0). Both tables are written on the device by Kernel_BLMainLaneRanges, once per round.
    const uint32_t* pRange;     // per-bin {firstLane, nLanes, tupleBase}, kMainRangeStride * kMainNBins
    const uint32_t* pBlockMap;  // per-block {bin, firstLane, endLane}, kMainBlockMapStride * kMainFusedBlocks
    WorkDiv1D workDivFused;     // kMainFusedBlocks x kFitBlock, fixed and host-known
  };

  // One (fast-fit + fit) launch of multiplicity N over the current chunk. nHitsL/nHitsH are the
  // multiplicity bin of the fast-fit (== N for exact bins, [maxHitsOnTrackForFullFit, maxHitsOnTrack-1]
  // for the "rest" bin). The CA main fit is the factorized fast BrokenLine fit: one launch, one
  // linearization, no phase split.
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
                        Kernel_BLFit<N, TrackerTraits>{c.rhoMap, c.bMap, c.fitCorrections},
                        c.tupleMultiplicity,
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pgblScratch);
  }

  // ---------------------------------------------------------------------------------------------
  // FUSED MAIN-LADDER LAUNCHERS. One launch per phase, all N-bins, on the fixed fused grid. Declared
  // `extern template` and instantiated ONE PHASE PER TU: a fused kernel pulls every compile-time N of
  // its phase into whichever TU instantiates it (per-phase rather than per-N build-time split).
  // ---------------------------------------------------------------------------------------------
  template <typename TrackerTraits>
  void runMainFusedFast(BLMainLaunchCtx<TrackerTraits> const& c) {
    BLMainFusedCfg cfg{};
    cfg.hasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFastFitFused<TrackerTraits>{cfg},
                        c.tuples,
                        c.tupleMultiplicity,
                        c.hv,
                        c.cm,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pRange,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runMainFusedFit(BLMainLaunchCtx<TrackerTraits> const& c) {
    BLMainFusedCfg cfg{};
    cfg.rhoMap = c.rhoMap;
    cfg.bMap = c.bMap;
    cfg.fitCorrections = c.fitCorrections;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitFused<TrackerTraits>{cfg},
                        c.tupleMultiplicity,
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pgblScratch,
                        c.pBlockMap);
  }

  // Explicit-instantiation signatures (expanded with `extern` below to suppress instantiation in the
  // orchestration TU, and bare in each BrokenLineFit_*.dev.cc to instantiate its disjoint N-subset).
  // The factorized fast BrokenLine main fit, the only fit the CA runs. The stubs N-range (3..10) is
  // compiled in BrokenLineFit_stubsMainBL.dev.cc; the four other traits (3..6) share
  // BrokenLineFit_main.dev.cc.
#define BLFIT_MAIN_SIG_BL(N, T)                    \
  template void runMainBin<N, ::pixelTopology::T>( \
      BLMainLaunchCtx<::pixelTopology::T> const&, WorkDiv1D const&, uint32_t, uint32_t, int32_t)
// The FUSED main ladder: one signature per PHASE and per traits set (each pulls every compile-time N
// of that phase into the TU that instantiates it).
#define BLFIT_MAIN_FUSED_FAST_SIG(T) \
  template void runMainFusedFast<::pixelTopology::T>(BLMainLaunchCtx<::pixelTopology::T> const&)
#define BLFIT_MAIN_FUSED_FIT_SIG(T) \
  template void runMainFusedFit<::pixelTopology::T>(BLMainLaunchCtx<::pixelTopology::T> const&)

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

  // Main fit, FUSED ladder: one instantiation per (phase, traits). The stubs phases get a TU each
  // (they carry N = 3..10); the four non-stubs traits (N = 3..6) share one.
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase1);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2OT);
  extern BLFIT_MAIN_FUSED_FAST_SIG(HIonPhase1);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2OTStubs);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase1);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2OT);
  extern BLFIT_MAIN_FUSED_FIT_SIG(HIonPhase1);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2OTStubs);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
