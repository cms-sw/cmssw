#ifndef RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h

// Device kernels of the BrokenLine fit: the per-lane bodies and the fused kernels that dispatch them over the N-bins.
// HelixFit::launchBrokenLineKernels (BrokenLineFit.h) launches them.

// #define BROKENLINE_DEBUG
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
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

  // Per round, one fast-fit launch and one fit launch cover all hit-count bins.
  // The bins share one set of L = riemannFit::maxNumberOfConcurrentFits lanes, cut into per-block tiles.
  // Round r fits the tuples [r L, (r + 1) L) of all bins, concatenated in bin order.
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
  // The task of one block of the fused grid: the lanes [firstLane, endLane) of bin `bin`.
  // bin == kFitNBins marks an idle block. The tiles before tile `tile` hold hitsBefore hits in all.
  struct BLBlockTask {
    uint32_t bin;
    uint32_t firstLane;
    uint32_t endLane;
    BLBinRange range;
    uint32_t tile;
    uint32_t hitsBefore;
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

  // The lane buffers (BLLaneBuffers) are cut into tiles, one per block of the fused grid.
  // In a tile, element e of lane j is at e * kFitLaneStride + j, so a warp reads consecutive addresses.
  inline constexpr uint32_t kFitLaneStride = kFitBlock;

  // The values of a lane buffer in one tile: perHit per hit of the tile's N, and perTile whatever N.
  struct BLTileValues {
    std::size_t perHit;
    std::size_t perTile;
    // the offset of the tile after `tiles` tiles that hold `hitsBefore` hits in all
    ALPAKA_FN_HOST_ACC constexpr std::size_t offset(uint32_t tiles, uint32_t hitsBefore) const {
      return perHit * hitsBefore + perTile * tiles;
    }
  };

  // The maps of one lane in its tile.
  template <int N>
  using BLMap3xNd = Eigen::Map<riemannFit::Matrix3xNd<N>, 0, Eigen::Stride<3 * kFitLaneStride, kFitLaneStride>>;
  template <int N>
  using BLMap6xNf = Eigen::Map<riemannFit::Matrix6xNf<N>, 0, Eigen::Stride<6 * kFitLaneStride, kFitLaneStride>>;
  using BLMap4d = Eigen::Map<riemannFit::Vector4d, 0, Eigen::InnerStride<kFitLaneStride>>;

  // The inputs of lane `lane`: the N hits, their six covariance entries and the pre-fit (X0, Y0, R, ds_T/dz).
  // The fast-fit lane writes them and the fit lane reads them.
  template <int N>
  struct LaneInputs {
    static constexpr std::size_t kHitValues = 3 * N, kGeValues = 6 * N, kFastFitValues = 4;
    BLMap3xNd<N> hits;
    BLMap6xNf<N> hitsGe;
    BLMap4d fastFit;
    ALPAKA_FN_HOST_ACC LaneInputs(double* phits, float* phitsGe, double* pfastFit, uint32_t lane)
        : hits(phits + lane), hitsGe(phitsGe + lane), fastFit(pfastFit + lane) {}
  };

  // The sensor records of one lane's hits for brokenline::coarseCoordinateRule. Column i is the HitSensor of hit i:
  // rows 0 and 1 of the sensor frame's rotation, then sigma_m^2 and sigma_s^2.
  inline constexpr int kHitSensorValues = 8;
  template <int N>
  struct HitSensorRecords {
    Eigen::Map<Eigen::Matrix<float, kHitSensorValues, N>,
               0,
               Eigen::Stride<kHitSensorValues * kFitLaneStride, kFitLaneStride>>
        values;
    ALPAKA_FN_ACC HitSensorRecords(float* block, uint32_t lane) : values(block + lane) {}
    // Write the frame and local variances of hit i. sig2s = 0 marks a hit that the rule does not take.
    ALPAKA_FN_ACC void set(int i, SOAFrame<float> const& frame, float sig2m, float sig2s) {
      auto const& r = frame.rotation();
      values.col(i) << r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(), sig2m, sig2s;
    }
    ALPAKA_FN_ACC brokenline::HitSensor sensor(int i) const {
      auto const& v = values.col(i);
      return brokenline::HitSensor{v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7)};
    }
  };

  // The partner records of one lane's hits: the second sensor of a vector hit (BrokenLine.h, combinePartners).
  // Column i holds the PartnerSensor and the PartnerMeasurement of hit i, in float.
  template <int N>
  struct PartnerRecords {
    static constexpr int kN = N;
    enum Row : int { kX, kY, kZ, kSig2m, kOffset, kWeight, kRows };
    Eigen::Map<Eigen::Matrix<float, kRows, N>, 0, Eigen::Stride<kRows * kFitLaneStride, kFitLaneStride>> values;
    ALPAKA_FN_ACC PartnerRecords(float* block, uint32_t lane) : values(block + lane) {}
    ALPAKA_FN_ACC brokenline::PartnerSensor partner(int i) const {
      return brokenline::PartnerSensor{values(kX, i), values(kY, i), values(kZ, i), values(kSig2m, i)};
    }
    ALPAKA_FN_ACC void setPartner(int i, brokenline::PartnerSensor const& q) {
      values(kX, i) = float(q.x);
      values(kY, i) = float(q.y);
      values(kZ, i) = float(q.z);
      values(kSig2m, i) = float(q.sig2m);
    }
    ALPAKA_FN_ACC brokenline::PartnerMeasurement measurement(int i) const {
      return brokenline::PartnerMeasurement{values(kOffset, i), values(kWeight, i)};
    }
    ALPAKA_FN_ACC void setMeasurement(int i, brokenline::PartnerMeasurement const& m) {
      values(kOffset, i) = float(m.offset);
      values(kWeight, i) = float(m.weight);
    }
  };
  inline constexpr int kPartnerValues = PartnerRecords<1>::kRows;

  // The lane buffers of the coarse-coordinate rule, for vector hits with fitCorrections. Per hit:
  //   sensors   the HitSensorRecords;
  //   hits      the hit and its covariance (hitsGe) as measured, for the line stage;
  //   moved     1 if the rule moved the hit, else 0;
  //   partners  the PartnerRecords.
  // One value per buffer: the buffer sizes, or the offsets of a tile.
  struct BLRuleValues {
    std::size_t sensors = 0;
    std::size_t hits = 0;
    std::size_t hitsGe = 0;
    std::size_t moved = 0;
    std::size_t partners = 0;
    // whether every value is at most the one in `sizes`
    ALPAKA_FN_HOST_ACC constexpr bool fitsIn(BLRuleValues const& sizes) const {
      return sensors <= sizes.sensors && hits <= sizes.hits && hitsGe <= sizes.hitsGe && moved <= sizes.moved &&
             partners <= sizes.partners;
    }
  };
  // The base pointers of the buffers; null where the buffers are not allocated.
  struct BLRulePointers {
    float* sensors = nullptr;
    double* hits = nullptr;
    float* hitsGe = nullptr;
    float* moved = nullptr;
    float* partners = nullptr;
  };

  // The view of lane `lane` of the tile at the offsets `tile`.
  template <int N>
  struct BLCoarseRuleLane {
    using MovedMap = Eigen::Map<Eigen::Matrix<float, 1, N>, 0, Eigen::InnerStride<kFitLaneStride>>;
    // the rules' view for circleFit and lineFitOnFittedCircle
    using Hits = brokenline::CoarseHitsLane<HitSensorRecords<N>, MovedMap, PartnerRecords<N>>;
    HitSensorRecords<N> sensors;
    BLMap3xNd<N> hits;
    BLMap6xNf<N> hitsGe;
    MovedMap moved;
    PartnerRecords<N> partners;
    ALPAKA_FN_ACC BLCoarseRuleLane(BLRulePointers const& base, BLRuleValues const& tile, uint32_t lane)
        : sensors(base.sensors + tile.sensors, lane),
          hits(base.hits + tile.hits + lane),
          hitsGe(base.hitsGe + tile.hitsGe + lane),
          moved(base.moved + tile.moved + lane),
          partners(base.partners + tile.partners, lane) {}
    ALPAKA_FN_ACC Hits view() const { return Hits{sensors, moved, partners}; }
  };

  template <typename TrackerTraits>
  struct BLCoarseRuleBuffers {
    static constexpr bool kActive = std::is_same_v<TrackerTraits, pixelTopology::Phase2OTVectorHits>;
    // whether a fit call uses the buffers
    ALPAKA_FN_HOST_ACC static constexpr bool used(bool fitCorrections) { return kActive && fitCorrections; }
    // the offset in each buffer of the tile after `tiles` tiles whose lanes hold `hitsBefore` hits in all
    ALPAKA_FN_HOST_ACC static constexpr BLRuleValues at(uint32_t tiles, uint32_t hitsBefore) {
      using PerHit = LaneInputs<1>;  // the values of a hit and of its covariance
      return BLRuleValues{BLTileValues{kHitSensorValues * kFitLaneStride, 0}.offset(tiles, hitsBefore),
                          BLTileValues{PerHit::kHitValues * kFitLaneStride, 0}.offset(tiles, hitsBefore),
                          BLTileValues{PerHit::kGeValues * kFitLaneStride, 0}.offset(tiles, hitsBefore),
                          BLTileValues{kFitLaneStride, 0}.offset(tiles, hitsBefore),
                          BLTileValues{kPartnerValues * kFitLaneStride, 0}.offset(tiles, hitsBefore)};
    }
    template <int N>
    using Lane = BLCoarseRuleLane<N>;
  };

  // Where the fast-fit lane writes the sensor and partner records of its hits. Null buffers: no records.
  // otRecHits holds the partner rechits.
  struct BLSensorRecordsOut {
    BLRulePointers buffers;
    BLRuleValues tile;
    ::reco::OTRecHitsConstView otRecHits;
  };

  template <int N>
  class Kernel_BLFastFit {
  public:
    // The per-lane body of the fast fit. With a BLSensorRecordsOut (vector hits only), also write the hit records.
    template <typename HitsView, typename... TRecordsOut>
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
                                                    uint32_t nHitsH,
                                                    TRecordsOut const&... recordsOut) {
      static_assert(sizeof...(TRecordsOut) <= 1 && (std::is_same_v<TRecordsOut, BLSensorRecordsOut> && ...),
                    "one BLSensorRecordsOut at most");
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
        // Write the sensor record. sigma_s^2 = 0 for a pixel hit.
        // For a vector hit, write the partner record: the rechit on the other sensor of the stack.
        // A hit without a valid partner gets sig2m = 0.
        if constexpr (sizeof...(TRecordsOut) == 1) {
          BLSensorRecordsOut const& out = (recordsOut, ...);
          if (out.buffers.sensors != nullptr) {
            BLCoarseRuleLane<N> rule(out.buffers, out.tile, local_idx);
            rule.sensors.set(int(i), frame, hh[hit].xerrLocal(), isOTEntry(hh, int32_t(hit)) ? yerrFit : 0.f);
            brokenline::PartnerSensor partner{0., 0., 0., 0.};
            if (isVectorHit(hh, int32_t(hit))) {
              auto const vh = hh.vectorHit(int32_t(hit));
              const uint32_t pos = vh.posHitIdx(), lower = vh.lowerHitIdx(), upper = vh.upperHitIdx();
              const uint32_t other = (pos == lower) ? upper : ((pos == upper) ? lower : ~uint32_t(0));
              auto const& ot = out.otRecHits;
              if (other < uint32_t(ot.metadata().size())) {
                auto const rechit = ot[int32_t(other)];
                partner =
                    brokenline::PartnerSensor{rechit.xGlobal(), rechit.yGlobal(), rechit.zGlobal(), rechit.xerrLocal()};
              }
            }
            rule.partners.setPartner(int(i), partner);
          }
        }
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
    using CoarseRule = BLCoarseRuleBuffers<TrackerTraits>;

    // The per-lane body of the fit. It fits lane local_idx and writes the SoA row ptkids[local_idx].
    // Only vector hits with fitCorrections read ruleBuffers and ruleTile.
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
                                                    BLRulePointers ruleBuffers,
                                                    BLRuleValues ruleTile,
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
      static_assert(int(N) <= int(TrackerTraits::maxHitsOnTrackForFullFit), "a lane fits at most the top bin's N hits");
      // The O(N) state of the fit is in this lane's slice of pscratch.
      brokenline::LaneScratch<N, kFitLaneStride> scratch(pscratch, local_idx);
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
      // The dip-angle steps of the line on the pre-fit arcs, which circleFit couples to the curvature.
      if (fitCorrections) {
        if (!data.failed) {
          brokenline::lineOnPreFitArcs(acc, hits_ge, fast_fit, data, fitWs);
        }
        brokenline::lineDipSteps(data, fitWs);
      }
      // Vector hits with fitCorrections: the circle stage takes the moved hits, the line stage the hits as measured.
      if constexpr (CoarseRule::kActive) {
        if (fitCorrections) {
          moveCoarseCoordinates(acc, local_idx, ruleBuffers, ruleTile, hits, hits_ge, fast_fit, data, fitWs);
          typename CoarseRule::template Lane<N> rule(ruleBuffers, ruleTile, local_idx);
          brokenline::partnerMeasurements(acc, fast_fit, rule.sensors, rule.partners);
          brokenline::circleFit(
              acc, hits, hits_ge, fast_fit, bFieldEff, data, circle, estimate, fitWs, ctx, rule.view());
        } else {
          brokenline::circleFit(acc, hits, hits_ge, fast_fit, bFieldEff, data, circle, estimate, fitWs, ctx);
        }
      } else {
        brokenline::circleFit(acc, hits, hits_ge, fast_fit, bFieldEff, data, circle, estimate, fitWs, ctx);
      }
      if (fitCorrections) {
        if constexpr (CoarseRule::kActive) {
          const typename CoarseRule::template Lane<N> rule(ruleBuffers, ruleTile, local_idx);
          brokenline::lineFitOnFittedCircle(acc,
                                            rule.hits,
                                            rule.hitsGe,
                                            fast_fit,
                                            bFieldEff,
                                            data,
                                            circle,
                                            estimate,
                                            line,
                                            fitWs,
                                            cov5,
                                            ctx,
                                            rule.view());
        } else {
          brokenline::lineFitOnFittedCircle(
              acc, hits, hits_ge, fast_fit, bFieldEff, data, circle, estimate, line, fitWs, cov5, ctx);
        }
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
    // Apply the coarse-coordinate rule to one lane (BrokenLine.h, coarseCoordinateRule).
    // Keep a copy of the hits as measured for the line stage. Project each partner along the track.
    // If a hit moves, recompute the pre-fit and its geometry, and re-project the dip-angle row.
    template <typename M3xN, typename M6xN, typename V4, typename TData, typename TWs>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void moveCoarseCoordinates(Acc1D const& acc,
                                                                     uint32_t local_idx,
                                                                     BLRulePointers const& ruleBuffers,
                                                                     BLRuleValues const& ruleTile,
                                                                     M3xN& hits,
                                                                     M6xN& hits_ge,
                                                                     V4& fast_fit,
                                                                     TData& data,
                                                                     TWs& fitWs) {
      typename CoarseRule::template Lane<N> rule(ruleBuffers, ruleTile, local_idx);
      rule.hits = hits;
      rule.hitsGe = hits_ge;
      int nCandidates = 0;
      for (int i = 0; i < int(N); i++) {
        rule.moved(0, i) = 0.f;
        const brokenline::HitSensor sensor = rule.sensors.sensor(i);
        brokenline::PartnerSensor partner = rule.partners.partner(i);
        if (!data.failed && brokenline::hasPartner(partner)) {
          brokenline::projectPartner(
              acc, fast_fit, data, fitWs, i, hits(0, i), hits(1, i), hits(2, i), sensor, partner);
          rule.partners.setPartner(i, partner);
        }
        nCandidates += brokenline::isCoarseCandidate(sensor) ? 1 : 0;
      }
      if (nCandidates == 0 || data.failed) {
        return;
      }
      if (brokenline::coarseCoordinateRule(
              acc, hits, hits_ge, rule.sensors, rule.partners, rule.moved, fast_fit, data, fitWs) == 0) {
        return;
      }
      const double slopeMeasured = -data.qCharge / fast_fit(3);
      brokenline::fastFit(acc, hits, fast_fit);
      brokenline::preFitGeometry(acc, hits, fast_fit, data);
      brokenline::reprojectDipShift(acc, slopeMeasured, -data.qCharge / fast_fit(3), fitWs);
    }

    // Publish q/pT = 0, which Kernel_classifyTracks rejects, the other parameters and the covariance NaN,
    // and chi2 = +infinity. edm::isNotFinite does not work under -Ofast with clang and hipcc.
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

  // The partition of one round's lanes into bins. Call f(b, range) for every bin, in bin order.
  // population(b) gives the bin sizes, on the device or on the host.
  ALPAKA_NO_HOST_ACC_WARNING
  template <typename TrackerTraits, typename P, typename F>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void forEachFitBinRangeOf(P const& population, uint32_t round, F&& f) {
    constexpr uint32_t kLanes = riemannFit::maxNumberOfConcurrentFits;
    const uint32_t seatedBefore = round * kLanes;  // the tuples of the concatenation seated by the earlier rounds
    uint32_t base = 0u;                            // the lanes of this round given to the bins before b
    uint32_t pos = 0u;                             // the position of bin b's first tuple in the concatenation
    for (uint32_t b = 0; b < kFitNBins<TrackerTraits>; ++b) {
      const uint32_t n = population(b);
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

  // The bin populations on the device, and on the host from the copy of the tuple-multiplicity offsets.
  template <typename TrackerTraits>
  struct BLDevicePopulations {
    TupleMultiplicity const* tupleMultiplicity;
    ALPAKA_FN_ACC uint32_t operator()(uint32_t b) const {
      return fitBinPopulation<TrackerTraits>(tupleMultiplicity, b);
    }
  };
  template <typename TrackerTraits>
  struct BLHostPopulations {
    const uint32_t* off;
    ALPAKA_FN_HOST_ACC uint32_t operator()(uint32_t b) const {
      const uint32_t lo = off[fitBinNHitsL(b)];
      const uint32_t hi = off[fitBinNHitsH<TrackerTraits>(b) + 1u];
      return hi > lo ? hi - lo : 0u;
    }
  };

  template <typename TrackerTraits, typename F>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void forEachFitBinRange(TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                                         uint32_t round,
                                                         F&& f) {
    forEachFitBinRangeOf<TrackerTraits>(BLDevicePopulations<TrackerTraits>{tupleMultiplicity}, round, f);
  }

  // The number of tiles of one round, and the sum of their N.
  struct BLRoundTiles {
    uint32_t tiles = 0u;
    uint32_t hits = 0u;
    // the blocks of a bin's lane range
    ALPAKA_FN_HOST_ACC static constexpr uint32_t blocks(BLBinRange const& range) {
      return (range.nLanes + kFitBlock - 1u) / kFitBlock;
    }
    ALPAKA_FN_HOST_ACC void operator()(uint32_t b, BLBinRange const& range) {
      const uint32_t nBlocks = blocks(range);
      tiles += nBlocks;
      hits += nBlocks * (kFitMinN + b);
    }
  };

  // The task of block `blk` in round `round`. Each bin's lanes are cut into blocks of kFitBlock lanes.
  // The blocks after the last bin are idle.
  template <typename TrackerTraits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE BLBlockTask fitBlockTask(TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                                          uint32_t round,
                                                          uint32_t blk) {
    constexpr uint32_t kNBins = kFitNBins<TrackerTraits>;
    BLBlockTask task{kNBins, 0u, 0u, BLBinRange{0u, 0u, 0u}, 0u, 0u};  // idle
    BLRoundTiles before;                                               // the tiles of the bins before b
    forEachFitBinRange<TrackerTraits>(tupleMultiplicity, round, [&](uint32_t b, BLBinRange const& range) {
      if (task.bin == kNBins && blk < before.tiles + BLRoundTiles::blocks(range)) {
        const uint32_t blockFirstLane = range.firstLane + (blk - before.tiles) * kFitBlock;
        const uint32_t binEndLane = range.firstLane + range.nLanes;
        task = BLBlockTask{b,
                           blockFirstLane,
                           (blockFirstLane + kFitBlock < binEndLane) ? (blockFirstLane + kFitBlock) : binEndLane,
                           range,
                           blk,
                           before.hits + (blk - before.tiles) * (kFitMinN + b)};
      }
      before(b, range);
    });
    ALPAKA_ASSERT_ACC(before.tiles <= kFitFusedBlocks<TrackerTraits>);  // sum_b ceil(n_b/kFitBlock) <= width
    return task;
  }

  // One value per lane buffer: the buffer sizes, or the offsets of a tile.
  struct BLLaneValues {
    std::size_t hits;
    std::size_t hitsGe;
    std::size_t fastFit;
    std::size_t scratch;
    BLRuleValues rule;  // the buffers of the rules (BLCoarseRuleBuffers), all 0 without them
  };

  // The tile values of each lane buffer.
  template <typename TrackerTraits>
  struct BLLaneTiles {
    using PerHit = LaneInputs<1>;  // the inputs of a lane per hit, and the pre-fit
    ALPAKA_FN_HOST_ACC static constexpr BLTileValues hits() {
      return BLTileValues{PerHit::kHitValues * kFitLaneStride, 0};
    }
    ALPAKA_FN_HOST_ACC static constexpr BLTileValues hitsGe() {
      return BLTileValues{PerHit::kGeValues * kFitLaneStride, 0};
    }
    ALPAKA_FN_HOST_ACC static constexpr BLTileValues fastFit() {
      return BLTileValues{0, PerHit::kFastFitValues * kFitLaneStride};
    }
    // kFitScratchDoubles<N> = kScratchPerHit N + kScratchPerLane (FitScratchLayout)
    static constexpr std::size_t kScratchPerHit = brokenline::kFitScratchDoubles<4> - brokenline::kFitScratchDoubles<3>;
    static constexpr std::size_t kScratchPerLane = brokenline::kFitScratchDoubles<3> - 3 * kScratchPerHit;
    static_assert(brokenline::kFitScratchDoubles<int(TrackerTraits::maxHitsOnTrackForFullFit)> ==
                      int(kScratchPerHit * TrackerTraits::maxHitsOnTrackForFullFit + kScratchPerLane),
                  "the fit scratch of a lane is linear in N");
    // the fit scratch
    ALPAKA_FN_HOST_ACC static constexpr BLTileValues scratch() {
      return BLTileValues{kScratchPerHit * kFitLaneStride, kScratchPerLane * kFitLaneStride};
    }
    // the offset in each buffer of the tile after `tiles` tiles that hold `hitsBefore` hits in all
    ALPAKA_FN_HOST_ACC static constexpr BLLaneValues at(uint32_t tiles, uint32_t hitsBefore, bool coarseRule) {
      return BLLaneValues{hits().offset(tiles, hitsBefore),
                          hitsGe().offset(tiles, hitsBefore),
                          fastFit().offset(tiles, hitsBefore),
                          scratch().offset(tiles, hitsBefore),
                          coarseRule ? BLCoarseRuleBuffers<TrackerTraits>::at(tiles, hitsBefore) : BLRuleValues{}};
    }
    // the tile of block `task` in each buffer: where it starts, and where it ends (where a tile after it would start)
    ALPAKA_FN_HOST_ACC static constexpr BLLaneValues begin(BLBlockTask const& task, bool coarseRule) {
      return at(task.tile, task.hitsBefore, coarseRule);
    }
    ALPAKA_FN_HOST_ACC static constexpr BLLaneValues end(BLBlockTask const& task, bool coarseRule) {
      return at(task.tile + 1u, task.hitsBefore + kFitMinN + task.bin, coarseRule);
    }
    // whether the tile of block `task` ends inside buffers of `sizes`
    ALPAKA_FN_HOST_ACC static constexpr bool inside(BLLaneValues const& sizes,
                                                    BLBlockTask const& task,
                                                    bool coarseRule) {
      const BLLaneValues e = end(task, coarseRule);
      return e.hits <= sizes.hits && e.hitsGe <= sizes.hitsGe && e.fastFit <= sizes.fastFit &&
             e.scratch <= sizes.scratch && e.rule.fitsIn(sizes.rule);
    }
  };

  // The sizes of the lane buffers for nRounds rounds: for each buffer, the largest size that a round needs.
  template <typename TrackerTraits>
  inline BLLaneValues blLaneSizes(const uint32_t* off, uint32_t nRounds, bool coarseRule) {
    using Tiles = BLLaneTiles<TrackerTraits>;
    BLLaneValues sizes{};
    for (uint32_t round = 0; round < nRounds; ++round) {
      BLRoundTiles roundTiles;
      forEachFitBinRangeOf<TrackerTraits>(BLHostPopulations<TrackerTraits>{off}, round, roundTiles);
      const BLLaneValues end = Tiles::at(roundTiles.tiles, roundTiles.hits, coarseRule);
      sizes.hits = std::max(sizes.hits, end.hits);
      sizes.hitsGe = std::max(sizes.hitsGe, end.hitsGe);
      sizes.fastFit = std::max(sizes.fastFit, end.fastFit);
      sizes.scratch = std::max(sizes.scratch, end.scratch);
      sizes.rule.sensors = std::max(sizes.rule.sensors, end.rule.sensors);
      sizes.rule.hits = std::max(sizes.rule.hits, end.rule.hits);
      sizes.rule.hitsGe = std::max(sizes.rule.hitsGe, end.rule.hitsGe);
      sizes.rule.moved = std::max(sizes.rule.moved, end.rule.moved);
      sizes.rule.partners = std::max(sizes.rule.partners, end.rule.partners);
    }
    return sizes;
  }

  // The fused kernels. Each block gets its bin and lanes from fitBlockTask and runs the lane body of that bin's N.
  // `sizes` are the buffer sizes from blLaneSizes.
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
                                  uint32_t round,
                                  BLRulePointers ruleBuffers,
                                  bool coarseRule,
                                  ::reco::OTRecHitsConstView otRecHits,
                                  BLLaneValues sizes) const {
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
        using Tiles = BLLaneTiles<TrackerTraits>;
        ALPAKA_ASSERT_ACC((Tiles::inside(sizes, task, coarseRule)));
        const BLLaneValues tile = Tiles::begin(task, coarseRule);
        typename caStructures::tindex_type* const tileTkids = ptkids + task.firstLane;
        double* const tileHits = phits + tile.hits;
        float* const tileHitsGe = phits_ge + tile.hitsGe;
        double* const tileFastFit = pfast_fit + tile.fastFit;
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = task.laneOf(uint32_t(el.local));
          if (local_idx >= task.endLane) {
            break;
          }
          const uint32_t tuple_idx = range.tupleOf(local_idx);
          forFitBin<TrackerTraits>(task.bin, [&](auto bin) {
            constexpr uint32_t b = decltype(bin)::value;
            if constexpr (BLCoarseRuleBuffers<TrackerTraits>::kActive) {
              const BLSensorRecordsOut recordsOut{ruleBuffers, tile.rule, otRecHits};
              Kernel_BLFastFit<b + kFitMinN>::lane(acc,
                                                   uint32_t(el.local),
                                                   tuple_idx,
                                                   foundNtuplets,
                                                   tupleMultiplicity,
                                                   hh,
                                                   cm,
                                                   tileTkids,
                                                   tileHits,
                                                   tileHitsGe,
                                                   tileFastFit,
                                                   fitBinNHitsL(b),
                                                   fitBinNHitsH<TrackerTraits>(b),
                                                   recordsOut);
            } else {
              Kernel_BLFastFit<b + kFitMinN>::lane(acc,
                                                   uint32_t(el.local),
                                                   tuple_idx,
                                                   foundNtuplets,
                                                   tupleMultiplicity,
                                                   hh,
                                                   cm,
                                                   tileTkids,
                                                   tileHits,
                                                   tileHitsGe,
                                                   tileFastFit,
                                                   fitBinNHitsL(b),
                                                   fitBinNHitsH<TrackerTraits>(b));
            }
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
                                  BLRulePointers ruleBuffers,
                                  uint32_t round,
                                  bool coarseRule,
                                  BLLaneValues sizes) const {
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
        using Tiles = BLLaneTiles<TrackerTraits>;
        ALPAKA_ASSERT_ACC((Tiles::inside(sizes, task, coarseRule)));
        const BLLaneValues tile = Tiles::begin(task, coarseRule);
        typename caStructures::tindex_type const* const tileTkids = ptkids + task.firstLane;
        double* const tileHits = phits + tile.hits;
        float* const tileHitsGe = phits_ge + tile.hitsGe;
        double* const tileFastFit = pfast_fit + tile.fastFit;
        double* const tileScratch = pscratch + tile.scratch;
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = task.laneOf(uint32_t(el.local));
          if (local_idx >= task.endLane) {
            break;
          }
          forFitBin<TrackerTraits>(task.bin, [&](auto bin) {
            constexpr uint32_t b = decltype(bin)::value;
            Kernel_BLFit<b + kFitMinN, TrackerTraits>::lane(acc,
                                                            uint32_t(el.local),
                                                            tupleMultiplicity,
                                                            bField,
                                                            results_view,
                                                            tileTkids,
                                                            tileHits,
                                                            tileHitsGe,
                                                            tileFastFit,
                                                            tileScratch,
                                                            ruleBuffers,
                                                            tile.rule,
                                                            cfg_.rhoMap,
                                                            cfg_.bMap,
                                                            cfg_.fitCorrections);
          });
        }
      }
    }
  };

  // The device lane buffers: the LaneInputs, the fit scratch and, only with coarseRule, the rule buffers.
  template <typename TrackerTraits>
  struct BLLaneBuffers {
    BLLaneBuffers(Queue& queue, BLLaneValues const& sizes, bool coarseRule)
        : hits(cms::alpakatools::make_device_buffer<double[]>(queue, sizes.hits)),
          hitsGe(cms::alpakatools::make_device_buffer<float[]>(queue, sizes.hitsGe)),
          fastFit(cms::alpakatools::make_device_buffer<double[]>(queue, sizes.fastFit)),
          scratch(cms::alpakatools::make_device_buffer<double[]>(queue, sizes.scratch)) {
      if (coarseRule) {
        rule.emplace(queue, sizes.rule);
      }
    }

    // The buffers of the coarse-coordinate rule.
    struct RuleBuffers {
      RuleBuffers(Queue& queue, BLRuleValues const& sizes)
          : sensors(cms::alpakatools::make_device_buffer<float[]>(queue, sizes.sensors)),
            hits(cms::alpakatools::make_device_buffer<double[]>(queue, sizes.hits)),
            hitsGe(cms::alpakatools::make_device_buffer<float[]>(queue, sizes.hitsGe)),
            moved(cms::alpakatools::make_device_buffer<float[]>(queue, sizes.moved)),
            partners(cms::alpakatools::make_device_buffer<float[]>(queue, sizes.partners)) {}

      cms::alpakatools::device_buffer<Device, float[]> sensors;
      cms::alpakatools::device_buffer<Device, double[]> hits;
      cms::alpakatools::device_buffer<Device, float[]> hitsGe;
      cms::alpakatools::device_buffer<Device, float[]> moved;
      cms::alpakatools::device_buffer<Device, float[]> partners;
    };

    // the base pointers of the rules' buffers, null without them
    BLRulePointers rulePointers() {
      if (!rule) {
        return BLRulePointers{};
      }
      return BLRulePointers{
          rule->sensors.data(), rule->hits.data(), rule->hitsGe.data(), rule->moved.data(), rule->partners.data()};
    }

    cms::alpakatools::device_buffer<Device, double[]> hits;
    cms::alpakatools::device_buffer<Device, float[]> hitsGe;
    cms::alpakatools::device_buffer<Device, double[]> fastFit;
    cms::alpakatools::device_buffer<Device, double[]> scratch;
    std::optional<RuleBuffers> rule;  // allocated only with coarseRule
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#undef BL_LANE_NOINLINE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
