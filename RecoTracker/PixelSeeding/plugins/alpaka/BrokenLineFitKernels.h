#ifndef RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h

// Device kernels of the BrokenLine fit: the per-lane bodies and the fused kernels that dispatch them over the N-bins.
// HelixFit::launchBrokenLineKernels (BrokenLineFit.h) launches them.

// #define BROKENLINE_DEBUG
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"

#include "HelixFit.h"

using OutputSoAView = reco::TrackSoAView;
using TupleMultiplicity = caStructures::GenericContainer;
using Tuples = caStructures::SequentialContainer;
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Block size of the fused kernels, in lanes.
  inline constexpr uint32_t kFitBlock = 32u;

  // ---------------------------------------------------------------------------------------------
  // The fused ladder: per round, one fused fast-fit and one fused fit launch over all N-bins, which share the per-lane
  // buffers through a bin <-> lane partition.
  //
  // The N-bins share one set of per-lane buffers (tkids, hits, hits_ge, fast_fit and the fit scratch,
  // riemannFit::maxNumberOfConcurrentFits lanes; element e of lane l at e * riemannFit::stride + l, so two lanes never
  // share an element whatever their N). tupleMultiplicity bins the tuples by hit count, so different bins hold
  // different tuples (begin(nHitsL)..end(nHitsH)) and write different SoA rows: each bin takes its own lane range of
  // the shared buffers, and all bins run in one launch per phase.
  //
  // Each block of the fused kernels computes the partition on the device from the bin populations (the demand of bin
  // b is end(nHitsH) - begin(nHitsL)) and takes its bin and lanes from it (fitBlockTask). One round seats at most
  // L = maxNumberOfConcurrentFits lanes, given out in bin order: round r seats the tuples [r L, (r + 1) L) of the
  // bins' tuples concatenated in bin order (forEachFitBinRange), so the round index is the only state carried from
  // round to round. The launcher (HelixFit::launchBrokenLineKernels) runs enough rounds to fit every tuple.
  inline constexpr uint32_t kFitMinN = 3u;
  template <typename TrackerTraits>
  inline constexpr uint32_t kFitNBins = uint32_t(TrackerTraits::maxHitsOnTrackForFullFit) - kFitMinN + 1u;

  // Width of the fused grid, in blocks. It covers every partition of the lanes into bins.
  template <typename TrackerTraits>
  inline constexpr uint32_t kFitFusedBlocks =
      riemannFit::maxNumberOfConcurrentFits / kFitBlock + kFitNBins<TrackerTraits>;

  // The lanes of one bin in one round: lanes [firstLane, firstLane + nLanes) fit the bin's tuples from firstTuple.
  struct BLBinRange {
    uint32_t firstLane;
    uint32_t nLanes;
    uint32_t firstTuple;
    // the index, in the bin's slice of tupleMultiplicity, of the tuple that lane `lane` fits
    ALPAKA_FN_ACC uint32_t tupleOf(uint32_t lane) const { return firstTuple + (lane - firstLane); }
  };
  // The task of one block of the fused grid: the lanes [firstLane, endLane) of bin `bin`, whose lanes in the round are
  // `range`; bin == kFitNBins marks an idle block.
  struct BLBlockTask {
    uint32_t bin;
    uint32_t firstLane;
    uint32_t endLane;
    BLBinRange range;
    // the lane of the block's element `element`
    ALPAKA_FN_ACC uint32_t laneOf(uint32_t element) const { return firstLane + element; }
  };

  // The hit-count bins: one bin per N = kFitMinN .. maxHitsOnTrackForFullFit - 1, and a top bin.
  // The top bin takes the hit counts [maxHitsOnTrackForFullFit, maxHitsOnTrack - 1] and fits
  // N = maxHitsOnTrackForFullFit of them.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t fitBinNHitsL(uint32_t b) { return kFitMinN + b; }
  template <typename TrackerTraits>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t fitBinNHitsH(uint32_t b) {
    return (b + 1u == kFitNBins<TrackerTraits>) ? (uint32_t(TrackerTraits::maxHitsOnTrack) - 1u) : (kFitMinN + b);
  }
  // The number of tuples in bin b.
  template <typename TrackerTraits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t fitBinPopulation(TupleMultiplicity const* tupleMultiplicity, uint32_t b) {
    const int32_t tot =
        int32_t(tupleMultiplicity->end(fitBinNHitsH<TrackerTraits>(b)) - tupleMultiplicity->begin(fitBinNHitsL(b)));
    return tot > 0 ? uint32_t(tot) : 0u;
  }
  // The number of tuples in all bins. off[k] is the number of tuples with fewer than k hits.
  template <typename TrackerTraits>
  inline uint32_t fittedPopulation(const uint32_t* off) {
    const uint32_t lo = off[kFitMinN];
    const uint32_t hi = off[TrackerTraits::maxHitsOnTrack];
    return (hi > lo) ? (hi - lo) : 0u;
  }
  // Call f(std::integral_constant<uint32_t, bin>{}), so that f gets the bin's N at compile time.
  // Do nothing for a bin past the last one.
  template <typename F, uint32_t... B>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void forFitBinOf(uint32_t bin, F&& f, std::integer_sequence<uint32_t, B...>) {
    (void)((bin == B && (f(std::integral_constant<uint32_t, B>{}), true)) || ...);
  }
  template <typename TrackerTraits, typename F>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void forFitBin(uint32_t bin, F&& f) {
    forFitBinOf(bin, f, std::make_integer_sequence<uint32_t, kFitNBins<TrackerTraits>>{});
  }

  // Keep the per-lane fit bodies out of line. noclone stops -fipa-cp-clone from making a copy per call.
#if defined(__CUDACC__)
#define BL_LANE_NOINLINE __noinline__
#elif defined(__clang__)
#define BL_LANE_NOINLINE __attribute__((noinline))
#elif defined(__GNUC__)
#define BL_LANE_NOINLINE __attribute__((noinline, noclone))
#else
#define BL_LANE_NOINLINE
#endif

  // Element e of lane l of a per-lane buffer is at e * riemannFit::stride + l: the lanes of a round never share an element.
  static_assert(riemannFit::stride >= riemannFit::maxNumberOfConcurrentFits);

  // One lane's inputs in the fits' per-lane buffers (the maps of HelixFit.h): the N hits as doubles, their six
  // covariance entries as floats and the pre-fit (X0, Y0, R, ds_T/dz), written by the fast-fit lane, read by the fit
  // lane.
  template <int N>
  struct LaneInputs {
    static constexpr std::size_t kHitValues = 3 * N, kGeValues = 6 * N, kFastFitValues = 4;
    riemannFit::Map3xNd<N> hits;
    riemannFit::Map6xNf<N> hitsGe;
    riemannFit::Map4d fastFit;
    ALPAKA_FN_HOST_ACC LaneInputs(double* phits, float* phitsGe, double* pfastFit, uint32_t lane)
        : hits(phits + lane), hitsGe(phitsGe + lane), fastFit(pfastFit + lane) {}
  };

  template <int N>
  class Kernel_BLFastFit {
  public:
    // The per-lane body of the fast fit. HitsView: HitsMultiView, or the CAHitsView facade of Phase2OTVectorHits.
    template <typename HitsView>
    ALPAKA_FN_ACC BL_LANE_NOINLINE static void lane(Acc1D const& acc,
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
                                                    uint32_t nHitsH) {
      constexpr uint32_t hitsInFit = N;
      // get it from the ntuple container (one to one to helix)
      auto tkid = *(tupleMultiplicity->begin(nHitsL) + tuple_idx);
      ALPAKA_ASSERT_ACC(tkid < foundNtuplets->nOnes());

      ptkids[local_idx] = tkid;

      auto nHits = foundNtuplets->size(tkid);

      ALPAKA_ASSERT_ACC(nHits >= nHitsL);
      ALPAKA_ASSERT_ACC(nHits <= nHitsH);

      LaneInputs<N> in(phits, phits_ge, pfast_fit, local_idx);
      auto& hits = in.hits;
      auto& hits_ge = in.hitsGe;
      auto& fast_fit = in.fastFit;

      // Prepare data structure
      auto const* hitId = foundNtuplets->begin(tkid);

      float incr = std::max(1.f, float(nHits) / float(hitsInFit));
      float n = 0;
      for (uint32_t i = 0; i < hitsInFit; ++i) {
        int j = int(n + 0.5f);  // round
        if (hitsInFit - 1 == i) {
          j = nHits - 1;  // force last hit to ensure max lever arm.
        }
        ALPAKA_ASSERT_ACC(j < int(nHits));
        n += incr;
        auto hit = hitId[j];
        float ge[6];

        // Propagate the local errors to the global frame through the sensor that carries the hit position.
        auto frame = cm.innerSensorFrame(hh[hit].detectorIndex());
        float yerrFit = hh[hit].yerrLocal();
        frame.toGlobal(hh[hit].xerrLocal(), 0, yerrFit, ge);

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
  };

  // The effective field of a track for the curvature to pT conversion: bField times the hit average of
  // B_bend = Bz - Br tan(lambda) cos(alpha), relative to Bz(0,0). tan(lambda) cos(alpha) comes from the pre-fit.
  template <typename TAcc, typename M3xN, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double blEffectiveBField(
      const TAcc& acc, const M3xN& hits, int n, const V4& fast_fit, double bField, const blBFieldMap::Map* bMap) {
    const double cx = fast_fit(0);
    const double cy = fast_fit(1);
    const double absR = alpaka::math::abs(acc, fast_fit(2));
    const double slopeDen = fast_fit(3) * absR;
    double sSum = 0.;
    for (int i = 0; i < n; ++i) {
      const double x = hits(0, i);
      const double y = hits(1, i);
      const double r = alpaka::math::sqrt(acc, x * x + y * y);
      sSum += blBFieldMap::bBendAt(*bMap, r, hits(2, i), brokenline::tanLambdaCosAlpha(cx, cy, slopeDen, x, y, r));
    }
    return bField * (sSum / double(n));
  }

  // The BrokenLine fit (circle and line) of one lane.
  template <int N, typename TrackerTraits>
  struct Kernel_BLFit {
  public:
    // The per-lane body of the fit: it fits lane local_idx and writes the one SoA row ptkids[local_idx] names.
    ALPAKA_FN_ACC BL_LANE_NOINLINE static void lane(Acc1D const& acc,
                                                    uint32_t local_idx,
                                                    TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                                    double bField,
                                                    OutputSoAView& results_view,
                                                    typename caStructures::tindex_type const* __restrict__ ptkids,
                                                    double* __restrict__ phits,
                                                    float* __restrict__ phits_ge,
                                                    double* __restrict__ pfast_fit,
                                                    double* __restrict__ pscratch,
                                                    const blMaterialMap::Map* rhoMap,
                                                    const blBFieldMap::Map* bMap,
                                                    bool fitCorrections) {
      auto tkid = ptkids[local_idx];

      ALPAKA_ASSERT_ACC(tkid < tupleMultiplicity->capacity());

      LaneInputs<N> in(phits, phits_ge, pfast_fit, local_idx);
      auto& hits = in.hits;
      auto& hits_ge = in.hitsGe;
      auto& fast_fit = in.fastFit;

      static_assert(TrackerTraits::maxHitsOnTrackForFullFit <= uint32_t(brokenline::kPivotMaxNodes),
                    "the pivot test of BrokenLine.h is counted for at most kPivotMaxNodes fitted hits");
      // The O(N) state of the fit is in this lane's slice of pscratch.
      static_assert(int(N) <= int(TrackerTraits::maxHitsOnTrackForFullFit),
                    "the fit scratch is sized for maxHitsOnTrackForFullFit hits");
      brokenline::LaneScratch<N, riemannFit::stride> scratch(pscratch, local_idx);
      auto& data = scratch.prepared;
      auto& fitWs = scratch.workspace;
      brokenline::karimaki_circle_fit circle;
      riemannFit::LineFit line;
      brokenline::CircleEstimate estimate;
      Eigen::Matrix<double, 5, 5> cov5;
      // The effective field converts the curvature to q/pT and sets the momentum for scattering and energy loss.
      const double bFieldEff = fitCorrections ? blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap) : bField;
      const brokenline::FitContext ctx{fitCorrections, rhoMap, bMap, bField};
      brokenline::prepareBrokenLineData(acc, hits, hits_ge, fast_fit, bFieldEff, data, fitWs, ctx);
      if (!fitCorrections) {
        brokenline::lineFit(acc, hits_ge, fast_fit, bFieldEff, data, line, fitWs, ctx);
      }
      brokenline::circleFit(acc, hits, hits_ge, fast_fit, bFieldEff, data, circle, estimate, fitWs, ctx);
      if (fitCorrections) {
        brokenline::lineFitOnFittedCircle(
            acc, hits, hits_ge, fast_fit, bFieldEff, data, circle, estimate, line, fitWs, cov5, ctx);
        brokenline::transportToFittedPca(acc, hits, data, bFieldEff, circle, line, ctx);
      }
      constexpr int8_t kNdof = 2 * N - 5;  // 2N measurements, 5 parameters
      if (fitCorrections) {
        // Convert kappa to q/pT = kappa / bFieldEff in the state and in the upper triangle of the covariance.
        const float invB = 1.f / float(bFieldEff);
        Eigen::Matrix<float, 5, 1> state;
        state << circle.par.template cast<float>(), line.par.template cast<float>();
        state(2) = float(circle.par(2)) * invB;
        Eigen::Matrix<float, 5, 5> stored = cov5.template cast<float>();
        for (int k = 0; k < 2; ++k) {
          stored(k, 2) = invB * float(cov5(k, 2));
        }
        for (int k = 3; k < 5; ++k) {
          stored(2, k) = invB * float(cov5(2, k));
        }
        stored(2, 2) = invB * invB * float(cov5(2, 2));
        // Publish a failed fit, or a covariance that is not positive definite, as failed.
        if (data.failed || !brokenline::publishable(state, stored)) {
          publishFailed(results_view, tkid, kNdof);
          return;
        }
        reco::copyFromDense(results_view, state, stored, tkid);
      } else {
        reco::copyFromCircle(results_view, circle.par, circle.cov, line.par, line.cov, 1.f / float(bFieldEff), tkid);
      }
      results_view[tkid].pt() = float(bFieldEff) / float(std::abs(circle.par(2)));
      results_view[tkid].eta() = alpaka::math::asinh(acc, line.par(0));
      results_view[tkid].chi2() = (circle.chi2 + line.chi2) / kNdof;
      results_view[tkid].ndof() = kNdof;
    }

  private:
    // The failed publication: NaN parameters and covariance, chi2 = +infinity.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void publishFailed(OutputSoAView& results_view,
                                                             caStructures::tindex_type tkid,
                                                             int8_t ndof) {
      constexpr float nan = std::numeric_limits<float>::quiet_NaN();
      // A bit pattern, because HIP's clang warns on numeric_limits<float>::infinity() under -Ofast.
      constexpr float inf = std::bit_cast<float>(0x7f800000u);
      reco::copyFromDense(
          results_view, Eigen::Matrix<float, 5, 1>::Constant(nan), Eigen::Matrix<float, 5, 5>::Constant(nan), tkid);
      results_view[tkid].state()(2) = 0.f;
      results_view[tkid].pt() = nan;
      results_view[tkid].eta() = nan;
      results_view[tkid].chi2() = inf;
      results_view[tkid].ndof() = ndof;
    }
  };

  // The partition of one round, computed where it is used from the bin populations and the round index. Bin b holds
  // the positions [pos_b, pos_b + n_b) of the bins' tuples concatenated in bin order; min(max(round * L - pos_b, 0),
  // n_b) of them were seated by earlier rounds (L = riemannFit::maxNumberOfConcurrentFits), and the bin takes as many
  // of the rest as the lanes left after the bins before it allow. f(b, range) is called for every bin in bin order.
  // ---------------------------------------------------------------------------------------------
  template <typename TrackerTraits, typename F>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void forEachFitBinRange(TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                                         uint32_t round,
                                                         F&& f) {
    constexpr uint32_t kLanes = riemannFit::maxNumberOfConcurrentFits;
    const uint32_t seatedBefore = round * kLanes;  // the tuples of the concatenation seated by the earlier rounds
    uint32_t base = 0u;                            // the lanes of this round given to the bins before b
    uint32_t pos = 0u;                             // the position of bin b's first tuple in the concatenation
    for (uint32_t b = 0; b < kFitNBins<TrackerTraits>; ++b) {
      const uint32_t n = fitBinPopulation<TrackerTraits>(tupleMultiplicity, b);
      const uint32_t seatedFromPos = seatedBefore > pos ? seatedBefore - pos : 0u;  // max(round * L - pos_b, 0)
      const uint32_t have = seatedFromPos < n ? seatedFromPos : n;                  // min(.., n_b)
      const uint32_t want = n - have;
      const uint32_t room = kLanes - base;
      const uint32_t got = (want < room) ? want : room;
      // the bin's first tuple this round seats is the first one not seated yet
      f(b, BLBinRange{base, got, have});
      base += got;
      pos += n;
    }
  }

  // The task of block `blk` of the fused grid in round `round`: the bins' lane ranges cut into consecutive blocks of
  // kFitBlock lanes, bin after bin (the last block of a bin may be partial); the blocks past the last bin are idle.
  // sum_b ceil(n_b / kFitBlock) <= kFitFusedBlocks (see kFitFusedBlocks), so every lane has a block.
  template <typename TrackerTraits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE BLBlockTask fitBlockTask(TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                                          uint32_t round,
                                                          uint32_t blk) {
    constexpr uint32_t kNBins = kFitNBins<TrackerTraits>;
    BLBlockTask task{kNBins, 0u, 0u, BLBinRange{0u, 0u, 0u}};  // idle
    uint32_t firstBlock = 0u;                                  // the first block of bin b
    forEachFitBinRange<TrackerTraits>(tupleMultiplicity, round, [&](uint32_t b, BLBinRange const& range) {
      const uint32_t nBlocks = (range.nLanes + kFitBlock - 1u) / kFitBlock;
      if (task.bin == kNBins && blk < firstBlock + nBlocks) {
        const uint32_t blockFirstLane = range.firstLane + (blk - firstBlock) * kFitBlock;
        const uint32_t binEndLane = range.firstLane + range.nLanes;
        task = BLBlockTask{b,
                           blockFirstLane,
                           (blockFirstLane + kFitBlock < binEndLane) ? (blockFirstLane + kFitBlock) : binEndLane,
                           range};
      }
      firstBlock += nBlocks;
    });
    ALPAKA_ASSERT_ACC(firstBlock <= kFitFusedBlocks<TrackerTraits>);  // sum_b ceil(n_b/kFitBlock) <= width
    return task;
  }

  // The fused kernels: each block takes its bin and lanes from fitBlockTask and runs the out-of-line lane
  // of that bin's N (Kernel_BLFastFit / Kernel_BLFit<b + kFitMinN>::lane); an idle block returns at once.
  template <typename TrackerTraits>
  struct Kernel_BLFastFitFused {
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  HitsMultiView hh,
                                  ::reco::CAModulesConstView cm,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  uint32_t round) const {
      // Entry assertions: the input buffers and containers are set.
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr uint32_t kNBins = kFitNBins<TrackerTraits>;
      constexpr uint32_t kTotal = kFitFusedBlocks<TrackerTraits> * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const BLBlockTask task = fitBlockTask<TrackerTraits>(tupleMultiplicity, round, blk);
        if (task.bin >= kNBins) {
          continue;  // idle block: this round's partition did not need it
        }
        const BLBinRange range = task.range;
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = task.laneOf(uint32_t(el.local));
          if (local_idx >= task.endLane) {
            break;
          }
          const uint32_t tuple_idx = range.tupleOf(local_idx);
          forFitBin<TrackerTraits>(task.bin, [&](auto bin) {
            constexpr uint32_t b = decltype(bin)::value;
            Kernel_BLFastFit<b + kFitMinN>::lane(acc,
                                                 local_idx,
                                                 tuple_idx,
                                                 foundNtuplets,
                                                 tupleMultiplicity,
                                                 hh,
                                                 cm,
                                                 ptkids,
                                                 phits,
                                                 phits_ge,
                                                 pfast_fit,
                                                 fitBinNHitsL(b),
                                                 fitBinNHitsH<TrackerTraits>(b));
          });
        }
      }
    }
  };

  template <typename TrackerTraits>
  struct Kernel_BLFitFused {
    BLFitConfig cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  double* __restrict__ pscratch,
                                  uint32_t round) const {
      ALPAKA_ASSERT_ACC(results_view.pt().data());
      ALPAKA_ASSERT_ACC(results_view.eta().data());
      ALPAKA_ASSERT_ACC(results_view.chi2().data());
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr uint32_t kNBins = kFitNBins<TrackerTraits>;
      constexpr uint32_t kTotal = kFitFusedBlocks<TrackerTraits> * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const BLBlockTask task = fitBlockTask<TrackerTraits>(tupleMultiplicity, round, blk);
        if (task.bin >= kNBins) {
          continue;  // idle block
        }
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = task.laneOf(uint32_t(el.local));
          if (local_idx >= task.endLane) {
            break;
          }
          forFitBin<TrackerTraits>(task.bin, [&](auto bin) {
            constexpr uint32_t b = decltype(bin)::value;
            Kernel_BLFit<b + kFitMinN, TrackerTraits>::lane(acc,
                                                            local_idx,
                                                            tupleMultiplicity,
                                                            bField,
                                                            results_view,
                                                            ptkids,
                                                            phits,
                                                            phits_ge,
                                                            pfast_fit,
                                                            pscratch,
                                                            cfg_.rhoMap,
                                                            cfg_.bMap,
                                                            cfg_.fitCorrections);
          });
        }
      }
    }
  };

  // The device buffers of the ladder's lanes beside the track ids, sized for the top bin's N =
  // maxHitsOnTrackForFullFit: the inputs of LaneInputs (hits, their covariances, the pre-fit) and the fit's solver
  // scratch (BrokenLine.h, FitScratchLayout), all at the element stride riemannFit::stride.
  template <typename TrackerTraits>
  struct BLLaneBuffers {
    static constexpr int kMaxN = int(TrackerTraits::maxHitsOnTrackForFullFit);
    using Inputs = LaneInputs<kMaxN>;

    explicit BLLaneBuffers(Queue& queue)
        : hits(cms::alpakatools::make_device_buffer<double[]>(queue, Inputs::kHitValues * riemannFit::stride)),
          hitsGe(cms::alpakatools::make_device_buffer<float[]>(queue, Inputs::kGeValues * riemannFit::stride)),
          fastFit(cms::alpakatools::make_device_buffer<double[]>(queue, Inputs::kFastFitValues * riemannFit::stride)),
          scratch(cms::alpakatools::make_device_buffer<double[]>(
              queue, brokenline::kFitScratchDoubles<kMaxN> * riemannFit::stride)) {}

    cms::alpakatools::device_buffer<Device, double[]> hits;
    cms::alpakatools::device_buffer<Device, float[]> hitsGe;
    cms::alpakatools::device_buffer<Device, double[]> fastFit;
    cms::alpakatools::device_buffer<Device, double[]> scratch;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#undef BL_LANE_NOINLINE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
