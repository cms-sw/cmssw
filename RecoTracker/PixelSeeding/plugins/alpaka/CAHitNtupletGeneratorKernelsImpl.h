#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS
// Per-track printf of the fitted chi2 and its inputs, for fit-quality calibration. Off -- the
// #define is commented out -- and it must stay off in any timed or high-occupancy run: the printf
// is per track.
// #define CA_CHI2_DUMP

// C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

// Alpaka includes
#include <alpaka/alpaka.hpp>

// CMSSW includes
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"
// Type defined unconditionally (zero memory); the TripletDumpSoAView kernel arg + writes are #ifdef'd.
#include "RecoTracker/PixelSeeding/interface/TripletDumpSoA.h"
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "RecoTracker/PixelSeeding/interface/CATrackFeatures.h"
#include "RecoTracker/PixelSeeding/interface/CAStubMS.h"
#include "CAFitHitSelection.h"

// local includes
#include "CACell.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAStructures.h"
#include "CATrackDNN.h"
#include "CATripletCuts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels {

  using namespace ::caStructures;

  constexpr uint32_t tkNotFound = std::numeric_limits<uint32_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  // Gate width of the two-parameter (1/pT, cot(theta)) compatibility check used by the Phase-1
  // specializations of the duplicate removers. Hard-wired, as upstream.
  constexpr float nSigma2Phase1 = 25.f;
  // The gate width nSigma^2 of the five-parameter compatibility check (all the other topologies) is
  // a runtime cfi parameter (AlgoParams::fastDupNSigma2_), reaching Kernel_fastDuplicateRemover and
  // Kernel_rejectDuplicate as the fastDupNSigma2 argument rather than a constant here. Its default
  // reproduces the constant upstream uses (5.f).
  constexpr int nTrackParameters = 5;
  // map: index of a track parameter -> index of its covariance
  HOST_DEVICE_CONSTANT std::array<uint8_t, nTrackParameters> iParam2iCov = {0u, 5u, 9u, 12u, 14u};

  // all of these below are mostly to avoid carrying around the relative namespace

  using Quality = ::pixelTrack::Quality;
  using TkSoAView = ::reco::TrackSoAView;
  using TkHitSoAView = ::reco::TrackHitSoAView;

  template <typename TrackerTraits>
  using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;

  using Counters = caHitNtupletGenerator::Counters;
  using HitToTuple = caStructures::GenericContainer;
  using HitContainer = caStructures::SequentialContainer;
  using TupleMultiplicity = caStructures::GenericContainer;
  using HitToCell = caStructures::GenericContainer;
  using CellToCell = caStructures::GenericContainer;
  using CellToTrack = caStructures::GenericContainer;

  using namespace cms::alpakatools;

  // Templated on the module-start view: the ModulesMultiView, or the CAHitsView facade, which
  // answers moduleStartOf() out of the pixel and stub module blocks.
  template <typename ModulesView>
  class SetHitsLayerStart {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ModulesView &mm,
                                  const reco::CALayersSoAConstView &ll,
                                  uint32_t *__restrict__ hitsLayerStart) const {
      ALPAKA_ASSERT_ACC(0 == caStructures::moduleStartOf(mm, 0));

      for (int32_t i : cms::alpakatools::uniform_elements(acc, ll.metadata().size())) {
        hitsLayerStart[i] = caStructures::moduleStartOf(mm, ll.layerStarts()[i]);
#ifdef GPU_DEBUG
        int old = i == 0 ? 0 : caStructures::moduleStartOf(mm, ll.layerStarts()[i - 1]);
        printf("LayerStart %d/%d at module %d: %d - %d\n",
               i,
               ll.metadata().size() - 1,
               ll.layerStarts()[i],
               hitsLayerStart[i],
               hitsLayerStart[i] - old);
#endif
      }
    }
  };

  template <typename HitsView>
  class Kernel_printSizes {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsView hh,
                                  TkSoAView tt,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks) const {
      if (cms::alpakatools::once_per_grid(acc))
        printf(
            "nSizes: hh.size() %d; hh.size() - hh.view(0).offsetBPIX2() %d; nCells %d; nTrips %d; "
            "nCellTracks %d; nTracks %d; tt.metadata().size() %d\n",
            static_cast<int>(hh.size()),
            static_cast<int>(hh.size()) - hh.view(0).offsetBPIX2(),
            *nCells,
            *nTrips,
            *nCellTracks,
            tt.nTracks(),
            tt.metadata().size());
    }
  };

  template <typename TrackerTraits>
  class Kernel_checkOverflows {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity const *tupleMultiplicity,
                                  HitToTuple const *hitToTuple,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks,
                                  caStructures::CAPairSoAConstView cellCell,
                                  caStructures::CAPairSoAConstView cellTrack,
                                  int32_t nHits,
                                  uint32_t maxNumberOfDoublets,
                                  AlgoParams const &params,
                                  Counters *counters) const {
      auto &c = *counters;
      // counters once per event
      if (cms::alpakatools::once_per_grid(acc)) {
        alpaka::atomicAdd(acc, &c.nEvents, 1ull, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &c.nHits, static_cast<unsigned long long>(nHits), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &c.nCells, static_cast<unsigned long long>(*nCells), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(
            acc, &c.nTuples, static_cast<unsigned long long>(apc->get().first), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc,
                          &c.nFitTracks,
                          static_cast<unsigned long long>(tupleMultiplicity->size()),
                          alpaka::hierarchy::Blocks{});
      }

#ifdef NTUPLE_DEBUGS
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("number of found cells %d \n found tuples %d with total hits %d out of %d\n",
               *nCells,
               apc->get().first,
               apc->get().second,
               nHits);
        if (apc->get().first < tracks_view.metadata().size()) {
          ALPAKA_ASSERT_ACC(foundNtuplets->size(apc->get().first) == 0);
          ALPAKA_ASSERT_ACC(foundNtuplets->size() == apc->get().second);
        }
      }

      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) > TrackerTraits::maxHitsOnTrack)  // current real limit
          printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
        ALPAKA_ASSERT_ACC(foundNtuplets->size(idx) <= TrackerTraits::maxHitsOnTrack);
        for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
          ALPAKA_ASSERT_ACC(int(*ih) < nHits);
      }
#endif

      if (cms::alpakatools::once_per_grid(acc)) {
        // Count overflows into the per-container overflow counters (non-corrupting:
        // the build kernels already clamp; these counters surface the magnitude).
        if (apc->get().first >= uint32_t(tracks_view.metadata().size())) {
          printf("Tuples overflow\n");
          alpaka::atomicAdd(acc, &c.nTupleOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nCells >= maxNumberOfDoublets) {
          printf("Cells overflow\n");
          alpaka::atomicAdd(acc, &c.nCellOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nTrips >= uint32_t(cellCell.metadata().size())) {
          printf("Triplets overflow\n");
          alpaka::atomicAdd(acc, &c.nTripletOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nCellTracks >= uint32_t(cellTrack.metadata().size())) {
          printf("TracksToCell overflow\n");
          alpaka::atomicAdd(acc, &c.nCellTrackOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
      }

      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];
        if (thisCell.hasFishbone() && !thisCell.isKilled())
          alpaka::atomicAdd(acc, &c.nFishCells, 1ull, alpaka::hierarchy::Blocks{});
        if (thisCell.isKilled())
          alpaka::atomicAdd(acc, &c.nKilledCells, 1ull, alpaka::hierarchy::Blocks{});
        if (!thisCell.unused())
          alpaka::atomicAdd(acc, &c.nEmptyCells, 1ull, alpaka::hierarchy::Blocks{});
        if ((0 == hitToTuple->size(thisCell.inner_hit_id())) && (0 == hitToTuple->size(thisCell.outer_hit_id())))
          alpaka::atomicAdd(acc, &c.nZeroTrackCells, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Always-on overflow sentinel (independent of doStats_, under which Kernel_checkOverflows runs).
  // The capacity guards in the build kernels truncate silently, so this kernel tests the same
  // conditions once per event and accumulates into a per-stream 8-word buffer owned by
  // CAHitNtupletGenerator, which reports any nonzero word at endStream:
  //   accum[0] = tuple-count overflow          (apc.first   >= tracks capacity)
  //   accum[1] = doublet/cell overflow         (nCells      >= maxNumberOfDoublets)
  //   accum[2] = cellToCell overflow           (nTriplets   >= cellCell capacity)
  //   accum[3] = cellToTrack overflow          (nCellTracks >= cellTrack capacity)
  //   accum[4] = hitContainer content overflow (apc.second  >  content slots)
  //   accum[5] = hitToTuple content overflow   (apc.second  >  its storage extent; UINT32_MAX
  //              disables the check); accum[6..7] reserved.
  // nCells / nTriplets / nCellTracks saturate (an index taken past the cap is given back), so
  // ">= cap" is the only reachable signature and also fires on an exactly-full event. apc is pure
  // demand (inc_add is never rolled back), which makes "> capacity" exact for the hit content and
  // ">= capacity" exact for the tuple count (Kernel_fillHitDetIndices caps ntracks at capacity-1).
  // hitToTuple has no capacity test of its own; apc.second is an upper bound on its demand, so
  // that check is a conservative alarm, disabled by the caller when the storage is sized from the
  // hits-in-tracks readback. Launched 1x1: one launch per event, no readback, no wait.
  class Kernel_overflowSentinel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  cms::alpakatools::AtomicPairCounter const *apc,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks,
                                  uint32_t tracksCap,
                                  uint32_t maxNumberOfDoublets,
                                  uint32_t cellCellCap,
                                  uint32_t cellTrackCap,
                                  uint32_t hitContentCap,
                                  uint32_t hitToTupleContentCap,
                                  uint32_t *__restrict__ accum) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        if (apc->get().first >= tracksCap)
          alpaka::atomicAdd(acc, &accum[0], 1u, alpaka::hierarchy::Blocks{});
        if (*nCells >= maxNumberOfDoublets)
          alpaka::atomicAdd(acc, &accum[1], 1u, alpaka::hierarchy::Blocks{});
        if (*nTrips >= cellCellCap)
          alpaka::atomicAdd(acc, &accum[2], 1u, alpaka::hierarchy::Blocks{});
        if (*nCellTracks >= cellTrackCap)
          alpaka::atomicAdd(acc, &accum[3], 1u, alpaka::hierarchy::Blocks{});
        if (apc->get().second > hitContentCap)
          alpaka::atomicAdd(acc, &accum[4], 1u, alpaka::hierarchy::Blocks{});
        if (hitToTupleContentCap != 0xFFFFFFFFu && apc->get().second > hitToTupleContentCap)
          alpaka::atomicAdd(acc, &accum[5], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fishboneCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view) const {
      constexpr auto reject = Quality::dup;

      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];
        if (!thisCell.isKilled())
          continue;

        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);
        for (auto i = 0u; i < cellTracksHisto->size(idx); i++)
          tracks_view[tracksOfCell[i]].quality() = reject;
      }
    }
  };

  // remove shorter tracks if sharing a cell
  // It does not seem to affect efficiency in any way!
  // Work division: Acc2D with Y indexing cells and X indexing warp lanes
  // (warpSize threads per cell). All lanes of a warp cooperate on a single cell
  template <typename TrackerTraits>
  class Kernel_earlyDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const &acc,
                                  CACell<TrackerTraits> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      constexpr auto reject = Quality::edup;  /// cannot be loose
      ALPAKA_ASSERT_ACC(nCells);

      const int32_t warpSize = alpaka::warp::getSize(acc);
      const int32_t laneId = static_cast<int32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u]);

      for (uint32_t idx : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
#ifdef CA_SIZES
        if (laneId == 0)
          printf("cellTracksSizes;%d;%d;%d\n", idx, cT.size(), cT.capacity());
#endif
        const int ntr = static_cast<int>(cellTracksHisto->size(idx));
        if (ntr < 2)
          continue;

        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);

        // Warp-reduce maxNl over the cell's tracks.
        // Lanes scan a strided subset of the cell's tracks and hold a local maxNl in register
        int32_t localMax = 0;
        for (int k = laneId; k < ntr; k += warpSize) {
          const int32_t nl = tracks_view[tracksOfCell[k]].nLayers();
          if (nl > localMax)
            localMax = nl;
        }
        // Warp-reduce to find the maxNl across all lanes. The result is uniform across the warp.
        // Idle lanes start with 0 and do not influence the result.
        // All lanes must be active for the shuffle to work: no branching or return early here.
        for (int32_t off = 1; off < warpSize; off <<= 1) {
          const int32_t y = alpaka::warp::shfl_xor(acc, localMax, off);
          if (y > localMax)
            localMax = y;
        }
        const int32_t maxNl = localMax;

        // Process tracks sequentially using warps
        for (int i = 0; i < ntr; ++i) {
          const auto it = tracksOfCell[i];
          const int32_t nli = tracks_view[it].nLayers();
          // Same nli and maxNl across lanes, so uniform check and no early return here to keep all lanes active.
          if (nli >= maxNl) {
            continue;
          }

          // Look for compatible tracks in the same cell with fewer layers and similar curvature
          // Mark as duplicate if both conditions are met.
          //
          // tracks_view[].pt() holds the PRE-FIT CURVATURE here, not a pT: CACell::find_ntuplets writes
          // `pt[it] = preCurvature` as the early reference this kernel compares. The demotion is terminal
          // (Kernel_fillMultiplicity skips Quality::edup: the loser is never fitted, classified or
          // converted), so the compatibility window is gated on the topology:
          //
          //  - Phase2OTStubs uses a window RELATIVE to the curvatures being compared. An absolute |dcurv|
          //    window is a growing fraction of the curvature as pT rises and above a few tens of GeV
          //    accepts EVERY pair whatever their momenta or charge, degenerating into "delete the shorter
          //    track on every shared cell" -- exactly what a high-pT jet core produces when a pixel
          //    cluster merged between two collimated tracks leaves one of them one layer short. The stub
          //    chain, whose long lever arm makes those cells common, is tuned with this form.
          //  - every other topology keeps the upstream absolute window, so the plain Phase-2 (and Phase-1
          //    fallback) pixel-track collections are unchanged with respect to the release.
          constexpr bool kHasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
          constexpr float kEarlyDupRelCurv = 0.05f;       // relative window, stub topology
          constexpr float kEarlyDupAbsCurv2 = 0.000001f;  // absolute |dcurv|^2 window, upstream
          // Same value as CACell<TrackerTraits>::kUninitializeCurvature for every topology.
          constexpr float kUninitCurv = std::numeric_limits<float>::max();
          const float curvi = tracks_view[it].pt();
          bool foundCompatible = false;
          // Parallelize inner loop across lanes
          for (int j = laneId; j < ntr; j += warpSize) {
            const auto jt = tracksOfCell[j];
            if (tracks_view[jt].nLayers() <= nli)
              continue;  // need a strictly longer companion
            const float curvj = tracks_view[jt].pt();
            const float dcurv = curvi - curvj;
            if constexpr (kHasStubs) {
              // An uninitialised pre-fit curvature (FLT_MAX) must never be compatible with anything;
              // skip it rather than let it through the relative window.
              if (curvi == kUninitCurv || curvj == kUninitCurv)
                continue;
              const float thr = kEarlyDupRelCurv * (std::abs(curvi) + std::abs(curvj));
              if (dcurv * dcurv <= thr * thr) {
                foundCompatible = true;
                break;
              }
            } else {
              if (dcurv * dcurv <= kEarlyDupAbsCurv2) {
                foundCompatible = true;
                break;
              }
            }
          }
          // All lanes converge here to check if any foundCompatible is true, and if so, mark track as duplicate.
          if (alpaka::warp::any(acc, static_cast<int32_t>(foundCompatible))) {
            // One thread assigns warp-wide decision
            if (laneId == 0) {
              tracks_view[it].quality() = reject;
            }
          }
        }
      }
    }
  };

  // Specialization for Phase-1 to keep the same behavior as before.
  // remove shorter tracks if sharing a cell
  // It does not seem to affect efficiency in any way!
  class Kernel_earlyDuplicateRemoverPhase1 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<pixelTopology::Phase1> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      constexpr auto reject = Quality::edup;  /// cannot be loose
      ALPAKA_ASSERT_ACC(nCells);
      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef CA_SIZES
        printf("cellTracksSizes;%d;%d;%d\n", idx, cT.size(), cT.capacity());
#endif
        if (cellTracksHisto->size(idx) < 2)
          continue;

        int8_t maxNl = 0;
        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);

        // find maxNl
        for (auto i = 0u; i < cellTracksHisto->size(idx); i++) {
          if (int(tracksOfCell[i]) > tracks_view.metadata().size())
            printf(">WARNING: %d %d %d %d\n", idx, i, int(tracksOfCell[i]), tracks_view.metadata().size());
          auto nl = tracks_view[tracksOfCell[i]].nLayers();
          maxNl = std::max(nl, maxNl);
        }

        // if (maxNl<4) continue;
        // quad pass through (leave it here for tests)
        //  maxNl = std::min(4, maxNl);

        for (auto i = 0u; i < cellTracksHisto->size(idx); i++) {
          auto it = tracksOfCell[i];

          if (int(it) > tracks_view.metadata().size())
            printf(">WARNING: %d %d %d\n", i, it, tracks_view.metadata().size());
          if (tracks_view[it].nLayers() < maxNl)
            tracks_view[it].quality() = reject;  // no race: simple assignment of the same constant
        }
      }
    }
  };

  // Order/backend-independent duplicate removal: a track's final quality must not depend on the order
  // concurrent threads run in. Two disciplines share the int32 quality scratch (device_qualityScratch_)
  // and the two helpers below:
  //   - Kernel_fastDuplicateRemover is cell-parallel (several threads may demote the same shared track):
  //     it reads quality(), accumulates demotions into the scratch via atomicMin, and Kernel_applyQuality
  //     copies the scratch back into quality()
  //   - The hit-based removers (rejectDuplicate, sharedHitCleaner, triplet/simpleTripletCleaner) are
  //     track-parallel single-writers: Kernel_snapshotQuality freezes quality() into the scratch, then
  //     each thread reads that snapshot and writes only its own track's quality() (no atomics, no copy-back)
  class Kernel_snapshotQuality {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t *__restrict__ qualityScratch) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes()))
        qualityScratch[i] = static_cast<int32_t>(tracks_view[i].quality());
    }
  };

  class Kernel_applyQuality {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes()))
        tracks_view[i].quality() = static_cast<Quality>(qualityScratch[i]);
    }
  };

  // ---------------------------------------------------------------------------------------------
  // Two-tier work division for Kernel_fastDuplicateRemover. The kernel does O(ntr^2) work per
  // cell (ntr = tracks through the cell), and ntr has a very long tail: almost every cell holds
  // at most one track, while one doublet in the core of a high-pT jet can be shared by thousands
  // of n-tuplets, so with one thread per cell the whole grid waits for that cell. Tier 1
  // (ntr <= kDupCoopMinTracks): one thread owns the cell and runs the serial loops. Tier 2
  // (ntr > kDupCoopMinTracks): the whole warp owns the cell, one heavy cell at a time, the lanes
  // discovering each other's heavy cells with one ballot per grid step. Both tiers are entered
  // from the same grid-stride loop, so there is no extra pass, launch or synchronisation. The
  // threshold sits well above the bulk of the distribution: the cooperative path costs the same
  // total work but pays it as warp time, so it only wins when one cell dominates its warp.
  //
  // FLOATING-POINT RULE: this file is built with -Ofast (BuildFile.xml: ofast-flag), i.e. with
  // -ffinite-math-only and -fassociative-math, under which the compiler may rewrite a negated
  // compare (!(a < b) -> a >= b, a different predicate when chi2() is NaN after a failed fit),
  // reassociate or contract, and do so differently in a different inlining context. To keep the
  // two tiers bit-identical, fastDupRemoverCell evaluates the same floating-point expressions in
  // the same order and positive form for both: score(it) is re-read inside the loop (not hoisted),
  // the compatibility test runs BEFORE the ordering test, only the loop headers (which thread owns
  // which i) differ, and the maxQual / min-chi2 passes are run redundantly by every lane so that
  // no reduction or shuffle touches a float. The cell body contains no warp collective at all.
  //
  // Convergence of the two integer collectives in the driver (alpaka issues them with the full
  // lane mask, so every lane must reach them): (a) the block size is a multiple of the warp size
  // (enforced in the launcher), so a warp holds consecutive, warp-aligned grid thread indices;
  // (b) the grid-stride loop runs up to round_up_by(*nCells, warpSize), so all lanes of a warp
  // have the same trip count, lanes beyond *nCells joining the ballot with ntr = 0. On the serial
  // and TBB CPU backends the warp size is 1: the ballot is the predicate itself, every shuffle
  // returns its own argument and (iFirst, iStep) constant-fold to (0, 1).
  inline constexpr int kDupCoopMinTracks = 64;

  // Per-cell body of Kernel_fastDuplicateRemover.
  //   Coop == false: (iFirst, iStep) = (0, 1)              -> serial loops, one thread per cell
  //   Coop == true : (iFirst, iStep) = (laneId, warpSize)  -> one warp per cell
  // Redistributing the i's cannot change the result: the kernel never writes tracks_view (every
  // read is of the frozen values Kernel_snapshotQuality captured, so no thread observes another's
  // demotion); the only writes are atomicMin(&qualityScratch[t], q) with q `reject` or `loose`,
  // and atomicMin is commutative and idempotent, so the final entry is the minimum over the SET of
  // demotions whatever their order; and the `break` after a demotion is only an early exit, since
  // any compatible better partner demotes `it` to the same `reject`.
  template <bool Coop>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fastDupRemoverCell(Acc1D const &acc,
                                                         CellToTrack const *__restrict__ cellTracksHisto,
                                                         TkSoAView tracks_view,
                                                         int32_t *__restrict__ qualityScratch,
                                                         uint32_t cellIdx,
                                                         int ntr,
                                                         int lane,
                                                         int stride,
                                                         Quality reject,
                                                         float fastDupNSigma2) {
    constexpr auto loose = Quality::loose;

    auto score = [&](uint32_t it) { return tracks_view[it].chi2(); };
    auto demote = [&](uint32_t it, Quality q) {
      alpaka::atomicMin(acc, &qualityScratch[it], static_cast<int32_t>(q), alpaka::hierarchy::Blocks{});
    };

    auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(cellIdx);

    // The i's this thread owns. Both are compile-time constants for tier 1, so the compiler sees
    // a plain `for (int i = 0; i < ntr; ++i)`.
    const int iFirst = Coop ? lane : 0;
    const int iStep = Coop ? stride : 1;

    // Demote any track dominated by a compatible, strictly better one (higher quality, or equal
    // quality and lower chi2); each track tests all others and exact ties keep both
    for (int i = iFirst; i < ntr; i += iStep) {
      auto it = thisCellTracks[i];
      auto qi = tracks_view[it].quality();
      if (qi <= reject)
        continue;

      // get track parameters and covariances
      float iParams[nTrackParameters];
      float iCovs[nTrackParameters];
      for (int p{0}; p < nTrackParameters; ++p) {
        iParams[p] = tracks_view[it].state()(p);
        iCovs[p] = tracks_view[it].covariance()(iParam2iCov[p]);
      }
      // function that compares the five track parameters of tracks it and jt
      auto incompatibleTrackParams = [&](uint32_t jt) -> bool {
        // comparing phi, tip, 1/pT, cotan(theta) and zip
        for (int p{0}; p < nTrackParameters; ++p) {
          const auto dpij = iParams[p] - tracks_view[jt].state()(p);
          const auto e2dpij = fastDupNSigma2 * (iCovs[p] + tracks_view[jt].covariance()(iParam2iCov[p]));
          if (dpij * dpij > e2dpij)
            return true;  // incompatible param found
        }
        return false;  // all params compatible
      };

      for (int j = 0; j < ntr; ++j) {
        if (j == i)
          continue;
        auto jt = thisCellTracks[j];
        auto qj = tracks_view[jt].quality();
        if (qj <= reject)
          continue;
        if (incompatibleTrackParams(jt))
          continue;
        if ((qj > qi) || (qj == qi && score(jt) < score(it))) {
          demote(it, reject);
          break;
        }
      }
    }

    // find maxQual -- run whole by every lane (no reduction: the value must come out of the same
    // code on every lane, and O(ntr) redundant integer loads are nothing against the pass above)
    auto maxQual = reject;  // no duplicate!
    for (int i = 0; i < ntr; i++) {
      auto q = tracks_view[thisCellTracks[i]].quality();
      if (q > maxQual)
        maxQual = q;
    }

    if (maxQual <= loose)
      return;  // warp-uniform when Coop: every lane ran the same loop over the same data

    // min chi2 among the best-quality tracks (read from the unmodified quality, which the dup-marking
    // above does not affect for the max-quality min-chi2 track) -- run whole by every lane, so mc
    // is bit-for-bit the same on every lane
    float mc = maxScore;
    for (int i = 0; i < ntr; i++) {
      auto it = thisCellTracks[i];
      if (tracks_view[it].quality() == maxQual && score(it) < mc)
        mc = score(it);
    }

    // mark all other duplicates (keep them loose) -- same test on every lane; only the WRITES are distributed
    for (int i = iFirst; i < ntr; i += iStep) {
      auto it = thisCellTracks[i];
      if (tracks_view[it].quality() > loose && score(it) > mc)
        demote(it, loose);
    }
  }

  // assume the above (so, short tracks already removed)
  // Work division: Acc1D, one cell per thread, with the whole warp ganging up on the rare cells
  // whose track list is longer than kDupCoopMinTracks. See the two-tier comment above.
  template <typename TrackerTraits>
  class Kernel_fastDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  int32_t *__restrict__ qualityScratch,
                                  bool dupPassThrough,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      ALPAKA_ASSERT_ACC(nCells);
      const uint32_t ntNCells = (*nCells);

      const int warpSize = static_cast<int>(alpaka::warp::getSize(acc));
      const int laneId = static_cast<int>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] % uint32_t(warpSize));
      // Invariant (a): the launcher must use a block size that is a multiple of the warp size.
      // (the extra parentheses keep the comma of the template argument list out of the macro call)
      ALPAKA_ASSERT_ACC((0u == alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] % uint32_t(warpSize)));
      // Invariant (b): lane-aligned extent, so a warp's lanes share their trip count.
      const uint32_t extent = cms::alpakatools::round_up_by(ntNCells, uint32_t(warpSize));

      for (auto idx : cms::alpakatools::uniform_elements(acc, extent)) {
        const bool inRange = (idx < ntNCells);
        const int ntr = inRange ? static_cast<int>(cellTracksHisto->size(idx)) : 0;

        // tier 2: hand the heavy cells of this warp to the whole warp, one at a time. The mask is
        // warp-uniform, so this loop and the two integer collectives inside it are convergent.
        auto heavyMask = alpaka::warp::ballot(acc, (ntr > kDupCoopMinTracks) ? 1 : 0);
        using MaskT = decltype(heavyMask);
        if (heavyMask) {
          for (int l = 0; l < warpSize; ++l) {
            if (MaskT{0} == ((heavyMask >> l) & MaskT{1}))
              continue;
            const uint32_t cell = static_cast<uint32_t>(alpaka::warp::shfl(acc, static_cast<int32_t>(idx), l));
            const int n = alpaka::warp::shfl(acc, ntr, l);
            fastDupRemoverCell<true>(
                acc, cellTracksHisto, tracks_view, qualityScratch, cell, n, laneId, warpSize, reject, fastDupNSigma2);
          }
        }

        // tier 1: one cell per thread, exactly as before
        if (inRange && ntr >= 2 && ntr <= kDupCoopMinTracks)
          fastDupRemoverCell<false>(
              acc, cellTracksHisto, tracks_view, qualityScratch, idx, ntr, 0, 1, reject, fastDupNSigma2);
      }
    }
  };

  // Phase-1 specialization
  template <>
  class Kernel_fastDuplicateRemover<pixelTopology::Phase1> {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<pixelTopology::Phase1> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  int32_t *__restrict__ qualityScratch,
                                  bool dupPassThrough,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      constexpr auto loose = Quality::loose;

      ALPAKA_ASSERT_ACC(nCells);
      const auto ntNCells = (*nCells);

      auto score = [&](uint32_t it) { return std::abs(reco::tip(tracks_view, it)); };
      auto demote = [&](uint32_t it, Quality q) {
        alpaka::atomicMin(acc, &qualityScratch[it], static_cast<int32_t>(q), alpaka::hierarchy::Blocks{});
      };

      for (auto idx : cms::alpakatools::uniform_elements(acc, ntNCells)) {
        int ntr = cellTracksHisto->size(idx);
        if (ntr < 2)
          continue;

        auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(idx);

        // Mark as duplicate any track dominated by a compatible, strictly better one
        // (order-independent; lower track index breaks exact ties)
        for (int i = 0; i < ntr; ++i) {
          auto it = thisCellTracks[i];
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;
          auto opi = tracks_view[it].state()(2);
          auto e2opi = tracks_view[it].covariance()(9);
          auto cti = tracks_view[it].state()(3);
          auto e2cti = tracks_view[it].covariance()(12);
          for (int j = 0; j < ntr; ++j) {
            if (j == i)
              continue;
            auto jt = thisCellTracks[j];
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            auto opj = tracks_view[jt].state()(2);
            auto ctj = tracks_view[jt].state()(3);
            auto dct = nSigma2Phase1 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto dop = nSigma2Phase1 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            if ((qj > qi) || (qj == qi && (score(jt) < score(it) || (score(jt) == score(it) && jt < it)))) {
              demote(it, reject);
              break;
            }
          }
        }

        // find maxQual
        auto maxQual = reject;  // no duplicate!
        for (int i = 0; i < ntr; i++) {
          auto q = tracks_view[thisCellTracks[i]].quality();
          if (q > maxQual)
            maxQual = q;
        }

        if (maxQual <= loose)
          continue;

        // keep the single best-quality, min-score track (lower index breaks ties); demote the rest
        float mc = maxScore;
        uint32_t im = tkNotFound;
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() == maxQual) {
            auto s = score(it);
            if (s < mc || (s == mc && it < im)) {
              mc = s;
              im = it;
            }
          }
        }

        if (tkNotFound == im)
          continue;

        // mark all other duplicates (keep them loose)
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() > loose && it != im)
            demote(it, loose);
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_connect {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc2D const &acc,
                                  cms::alpakatools::AtomicPairCounter *apc,  // just to zero them
                                  HitsMultiView hh,
                                  reco::CAGraphSoAConstView cc,
                                  reco::CATripletCutsSoAConstView tripletCuts,
                                  bool useTripletDNN,
                                  float tripletDNNThreshold,
#ifdef CA_TRIPLET_DUMP
                                  caStructures::TripletDumpSoAView tripletDump,  // per-triplet feature capture
#endif
                                  caStructures::CAPairSoAView cn,
                                  CACell<TrackerTraits> *cells,
                                  uint32_t const *nCells,
                                  uint32_t *nTrips,
                                  HitToCell const *__restrict__ outerHitHisto,
                                  CellToCell *cellNeighborsHisto,
                                  uint32_t *__restrict__ pipelineCounters) const {
      using Cell = CACell<TrackerTraits>;
      uint32_t maxTriplets = cn.metadata().size();

      if (cms::alpakatools::once_per_grid(acc)) {
        *apc = 0;
      }  // ready for next kernel

      // loop on outer cells
      for (uint32_t oCellIndex : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
        auto &outerCell = cells[oCellIndex];
        auto middleHitId = outerCell.inner_hit_id() - hh.view(0).offsetBPIX2();

        if (int(middleHitId) < 0)
          continue;

        auto const *__restrict__ outerHitCells = outerHitHisto->begin(middleHitId);
        auto const numberOfPossibleNeighbors = outerHitHisto->size(middleHitId);

        auto skips = cc[outerCell.layerPairId()].skipsLayers();

        // outer-cell values shared by all candidates
        auto const outer = TripletCuts<TrackerTraits>::makeOuter(acc, hh, tripletCuts, outerCell);

#ifdef CA_DEBUG
        printf("numberOfPossibleFromHisto;%d;%d;%d;%d;%d\n",
               *nCells,
               middleHitId,
               oCellIndex,
               outerCell.innerLayer(cc),
               numberOfPossibleNeighbors);
#endif

        // loop on inner cells
        for (uint32_t j : cms::alpakatools::independent_group_elements_x(acc, numberOfPossibleNeighbors)) {
          auto iCellIndex = outerHitCells[j];
          auto &innerCell = cells[iCellIndex];
          float curvature = 0.f;

          // apply compatibility cuts for this triplet (innerCell, outerCell); cc (CA layer-pair graph)
          // supplies the per-hit CA layer ids for the DNN layer-gap features and the CA_TRIPLET_DUMP row
#ifdef CA_TRIPLET_DUMP
          float dumpFeat[18] =
              {};  // accept() fills 18 BASE DNN features; written to SoA below (zero-init defense-in-depth)
          float dumpScore = -1.f;  // accept() fills the in-kernel DNN score (consistency check)
#endif
          if (TripletCuts<TrackerTraits>::accept(acc,
                                                 innerCell,
                                                 outer,
                                                 curvature,
                                                 hh,
                                                 tripletCuts,
                                                 cc,
                                                 useTripletDNN,
                                                 tripletDNNThreshold,
#ifdef CA_TRIPLET_DUMP
                                                 dumpFeat,
                                                 &dumpScore,
#endif
                                                 pipelineCounters)) {
            auto t_ind = alpaka::atomicAdd(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});

#ifdef CA_DEBUG
            printf("Triplet no. %d %.5f %.5f (%d %d) - %d %d -> (%d, %d, %d, %d) \n",
                   t_ind,
                   thetaCut,
                   dcaCut,
                   outerCell.layerPairId(),
                   innerCell.layerPairId(),
                   iCellIndex,
                   oCellIndex,
                   outerCell.inner_hit_id(),
                   outerCell.outer_hit_id(),
                   innerCell.inner_hit_id(),
                   innerCell.outer_hit_id());
            printf("filling cell no. %d %d: %d -> %d\n", t_ind, cellNeighborsHisto->size(), iCellIndex, oCellIndex);
#endif

            if (t_ind >= maxTriplets) {
#ifdef CA_WARNINGS
              printf("Warning!!!! Too many cell->cell (triplets) associations (limit = %d)!\n", cn.metadata().size());
#endif
              alpaka::atomicSub(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});
              break;
            }

#ifdef CA_TRIPLET_DUMP
            // Per-built-triplet training row: 18 BASE features (from accept) + the three merged-hit
            // indices (truth join key) + CA layers (layGap derived). t_ind < maxTriplets
            // guaranteed by the guard above; the SoA is sized like cn (tripletsN_).
            {
              auto row = tripletDump[t_ind];
              row.absCurvature() = dumpFeat[0];
              row.tipTimesCurvature() = dumpFeat[1];
              row.dca() = dumpFeat[2];
              row.curvatureStubs() = dumpFeat[3];
              row.curvatureStubsErrSquared() = dumpFeat[4];
              row.curvature13() = dumpFeat[5];
              row.dPhi12() = dumpFeat[6];
              row.dPhi13() = dumpFeat[7];
              row.dPhi23() = dumpFeat[8];
              row.dr12() = dumpFeat[9];
              row.dr13() = dumpFeat[10];
              row.r1() = dumpFeat[11];
              row.r2() = dumpFeat[12];
              row.r3() = dumpFeat[13];
              row.z1() = dumpFeat[14];
              row.z2() = dumpFeat[15];
              row.z3() = dumpFeat[16];
              row.nStubs() = dumpFeat[17];
              row.curvature() = curvature;  // SIGNED (Kernel_connect local, by-ref from accept); for derived feats
              row.lay1() = int32_t(innerCell.innerLayer(cc));
              row.lay2() = int32_t(outerCell.innerLayer(cc));
              row.lay3() = int32_t(outerCell.outerLayer(cc));
              row.h1() = uint32_t(innerCell.inner_hit_id());
              row.h2() = uint32_t(outerCell.inner_hit_id());
              row.h3() = uint32_t(outerCell.outer_hit_id());
              row.inKernelScore() = dumpScore;
            }
#endif

            // One bin per cell (bin = iCellIndex). The non-layer-skipping vs
            // layer-skipping distinction is encoded in bit 31 of the stored
            // outer-cell index:
            //   bit 31 = 0 -> non-layer-skipping neighbor
            //   bit 31 = 1 -> layer-skipping neighbor
            // Key-range guard. One bin per cell, and iCellIndex is a cell index below the cell
            // count the histogram was sized from, so this holds by construction; it is here so a
            // sizing mismatch drops the association instead of writing outside off[].
            if (iCellIndex < cellNeighborsHisto->nOnes())
              cellNeighborsHisto->count(acc, iCellIndex);

            cn[t_ind].inner() = iCellIndex;
            cn[t_ind].outer() = oCellIndex | (skips ? caStructures::kSkipsLayerFlag : 0u);
            outerCell.setStatusBits(Cell::StatusBit::kUsed);
            innerCell.setStatusBits(Cell::StatusBit::kUsed);

            // Pipeline stage counters: classify triplet by hit types
            if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
              if (pipelineCounters) {
                using PC = caHitNtupletGenerator::PipelineCounter;
                alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsTotal], 1u, alpaka::hierarchy::Blocks{});
                auto hit1 = innerCell.inner_hit_id();
                auto hit2 = outerCell.inner_hit_id();
                auto hit3 = outerCell.outer_hit_id();
                int nStubs = (isStub(hh, hit1) ? 1 : 0) + (isStub(hh, hit2) ? 1 : 0) + (isStub(hh, hit3) ? 1 : 0);
                if (nStubs == 0)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixPixPix], 1u, alpaka::hierarchy::Blocks{});
                else if (nStubs == 1)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixPixOT], 1u, alpaka::hierarchy::Blocks{});
                else if (nStubs == 2)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixOTOT], 1u, alpaka::hierarchy::Blocks{});
                else {
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOTOTOT], 1u, alpaka::hierarchy::Blocks{});
                  // OOO triplet region breakdown
                  auto layer1 = innerCell.innerLayer(cc);  // innermost
                  auto layer2 = outerCell.innerLayer(cc);  // middle
                  auto layer3 = outerCell.outerLayer(cc);  // outermost
                  bool l1Brl = (layer1 >= 28 && layer1 <= 33);
                  bool l2Brl = (layer2 >= 28 && layer2 <= 33);
                  bool l3Brl = (layer3 >= 28 && layer3 <= 33);
                  bool l1Fwd = (layer1 >= 34 && layer1 <= 43);  // disks at z > 0
                  bool l2Fwd = (layer2 >= 34 && layer2 <= 43);
                  bool l3Fwd = (layer3 >= 34 && layer3 <= 43);
                  bool l1Bwd = (layer1 >= 44 && layer1 <= 53);  // disks at z < 0
                  bool l2Bwd = (layer2 >= 44 && layer2 <= 53);
                  bool l3Bwd = (layer3 >= 44 && layer3 <= 53);
                  if (l1Brl && l2Brl && l3Brl)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_barrel], 1u, alpaka::hierarchy::Blocks{});
                  else if (l1Bwd && l2Bwd && l3Bwd)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_bwd], 1u, alpaka::hierarchy::Blocks{});
                  else if (l1Fwd && l2Fwd && l3Fwd)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_fwd], 1u, alpaka::hierarchy::Blocks{});
                  else if ((l1Brl || l2Brl) && (l2Bwd || l3Bwd))
                    alpaka::atomicAdd(
                        acc, &pipelineCounters[PC::kTripletsOOO_brlToBwd], 1u, alpaka::hierarchy::Blocks{});
                  else if ((l1Brl || l2Brl) && (l2Fwd || l3Fwd))
                    alpaka::atomicAdd(
                        acc, &pipelineCounters[PC::kTripletsOOO_brlToFwd], 1u, alpaka::hierarchy::Blocks{});
                  else
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_other], 1u, alpaka::hierarchy::Blocks{});
                }
              }
            }
          }
        }  // loop on inner cells
      }  // loop on outer cells
    }
  };

  template <typename TrackerTraits>
  class FillDoubletsHisto {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t *nCells,
                                  uint32_t offsetBPIX2,
                                  HitToCell *outerHitHisto,
                                  Counters *counters) const {
      const auto nKeys = outerHitHisto->nOnes();
      for (auto cellIndex : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef DOUBLETS_DEBUG
        printf("outerHitHisto;%d;%d\n", cellIndex, cells[cellIndex].outer_hit_id());
#endif
        auto const key = cells[cellIndex].outer_hit_id() - offsetBPIX2;
        // Key-range guard. The key space is one bin per outer hit, so a key past nOnes means the
        // hit->cell offsets were sized for a smaller hit count than the cells reference: drop the
        // association instead of writing outside off[]. Counted once per dropped association here;
        // the matching count pass (CAPixelDoubletsAlgos.h) skips exactly the same keys.
        if (key < nKeys)
          outerHitHisto->fill(acc, key, cellIndex);
        else
          alpaka::atomicAdd(acc, &counters->nHitToCellOverflow, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename CAPairView, typename Container>
  class Kernel_fillGenericPair {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CAPairView cn,
                                  uint32_t const *nElements,
                                  Container *genericHisto) const {
      const auto nKeys = genericHisto->nOnes();
      for (uint32_t index : cms::alpakatools::uniform_elements(acc, *nElements)) {
        auto const key = cn[index].inner();
        // Key-range guard, mirroring the count pass in Kernel_connect / CACell::find_ntuplets: the
        // key is a cell index below the cell count the histogram was sized from, so this holds by
        // construction and only a sizing mismatch can drop an entry here.
        if (key < nKeys)
          genericHisto->fill(acc, key, cn[index].outer());
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_find_ntuplets {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  const ::reco::CAGraphSoAConstView &cc,
                                  const ::reco::CANtupletCutsSoAConstView &ntupletCuts,
                                  TkSoAView tracks_view,
                                  HitContainer *foundNtuplets,
                                  CellToCell const *__restrict__ cellNeighborsHisto,
                                  CellToTrack *cellTracksHisto,
                                  caStructures::CAPairSoAView ct,
                                  CACell<TrackerTraits> *__restrict__ cells,
                                  uint32_t *nCellTracks,
                                  uint32_t const *nTriplets,
                                  uint32_t const *nCells,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  AlgoParams const &params) const {
      using Cell = CACell<TrackerTraits>;

#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc))
        printf("starting producing ntuplets from %d cells and %d triplets \n", *nCells, *nTriplets);
#endif

      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto const &thisCell = cells[idx];

        // cut by earlyFishbone
        if (thisCell.isKilled())
          continue;

        // we require at least three hits
        if (cellNeighborsHisto->size(idx) == 0)
          continue;

        // check if the layer pair of the cell is among the set of starting pairs
        auto pid = thisCell.layerPairId();
        bool doit = cc[pid].startingPair();

        // check if the most inner hit does not fulfill the starting requirement
        auto lid = thisCell.innerLayer(cc);
        if (thisCell.inner_r(hh) > ntupletCuts[lid].startMaxInnerR())
          doit = false;

        constexpr uint32_t maxDepth = TrackerTraits::maxLayersPerTrack - 1;
#ifdef CA_DEBUG
        printf(
            "LayerPairId %d and inner layer %d doit ? %d From cell %d with nNeighbors = %d and innerR=%f < "
            "maxInnerR=%f ?\n",
            pid,
            lid,
            doit,
            idx,
            cellNeighborsHisto->size(idx),
            thisCell.inner_r(hh),
            ntupletCuts[lid].startMaxInnerR());
#endif

        if (doit) {
          typename Cell::TmpTuple stack;
          // Per-thread buffer that find_ntuplets fills when it saves an ntuplet. Declared here, not inside the
          // recursive (fully inlined) find_ntuplets, so the stack holds one copy per thread, not one per depth.
          typename Cell::hindex_type hits[TrackerTraits::maxHitsOnTrack];

          stack.reset();
          thisCell.template find_ntuplets<maxDepth>(acc,
                                                    hh,
                                                    ntupletCuts,
                                                    cc,
                                                    cells,
                                                    *foundNtuplets,
                                                    cellNeighborsHisto,
                                                    cellTracksHisto,
                                                    nCellTracks,
                                                    ct,
                                                    *apc,
                                                    tracks_view.quality().data(),
                                                    tracks_view.nLayers().data(),
                                                    tracks_view.pt().data(),
                                                    stack,
                                                    hits,
                                                    params.minHitsPerNtuplet_);
          ALPAKA_ASSERT_ACC(stack.empty());
        }
      }
    }
  };
#ifdef CA_PIPELINE_COUNTERS
  // Pipeline counter: classify n-tuplets by OT hit content
  template <typename TrackerTraits>
  class Kernel_pipelineNtupletCount {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  cms::alpakatools::AtomicPairCounter const *apc,
                                  uint32_t maxTuples,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      using PC = caHitNtupletGenerator::PipelineCounter;
      // Clamp to container capacity -- apc may exceed maxTuples on overflow
      auto ntracks = std::min<uint32_t>(apc->get().first, maxTuples);
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        auto nh = foundNtuplets->size(idx);
        if (nh < 3)
          continue;
        alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsTotal], 1u, alpaka::hierarchy::Blocks{});
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          auto nHits = hh.size();
          int nOT = 0;
          for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
            if (*h >= static_cast<unsigned int>(nHits))
              break;  // content buffer corruption from overflow
            if (isStub(hh, *h))
              ++nOT;
          }
          if (nOT >= 1)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsWithOT], 1u, alpaka::hierarchy::Blocks{});
          if (nOT >= 3)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsOT3Plus], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Count cell status after all kill phases (reachability + fishbone)
  template <typename TrackerTraits>
  class Kernel_pipelineCellStatus {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *nCells,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      using PC = ::caHitNtupletGenerator::PipelineCounter;
      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &cell = cells[idx];
        if (!cell.unused())  // kUsed bit is set
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsUsedInTriplet], 1u, alpaka::hierarchy::Blocks{});
        if (cell.isKilled())
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsKilledTotal], 1u, alpaka::hierarchy::Blocks{});
        else
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsAlive], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Copy *nCellTracks into the pipeline counter array
  class Kernel_pipelineCopyCellTrackCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  uint32_t const *nCellTracks,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      if (cms::alpakatools::once_per_grid(acc))
        pipelineCounters[::caHitNtupletGenerator::kCellTrackPairs] = *nCellTracks;
    }
  };
#endif  // CA_PIPELINE_COUNTERS

  template <typename TrackerTraits>
  class Kernel_mark_used {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> *__restrict__ cells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  uint32_t const *nCells) const {
      using Cell = CACell<TrackerTraits>;
      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto &thisCell = cells[idx];
        if (cellTracksHisto->size(idx) > 0)
          thisCell.setStatusBits(Cell::StatusBit::kInTrack);
      }
    }
  };

  // Count the hits the fit will actually use, given the FitHitSelection mode
  // (== nhits in the default All mode). Shared by count/fillMultiplicity and kept
  // consistent with the fit's own selection in BrokenLineFit.dev.cc.
  template <typename TrackerTraits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t nSelectedHits(HitContainer const *__restrict__ foundNtuplets,
                                                        uint32_t it,
                                                        caStructures::HitsViewT<TrackerTraits> hh) {
    // hasStubs enables the OT-stub hit treatment (kMode filtering and the same-layer pixel overlap merge).
    // It must match the fit, which keys off the runtime offsetStubs: with no stubs, offsetStubs is the
    // unsigned sentinel and every hit is a plain pixel hit.
    const bool hasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits> &&
                          (static_cast<int32_t>(caStructures::offsetStubsOf(hh)) >= 0);
    return caFitHitSel::dedupWalk(foundNtuplets, it, hh, hasStubs, /*k=*/-1);
  }

  template <typename TrackerTraits>
  class Kernel_countMultiplicity {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);
        // printf("it: %d nhits: %d \n",it,nhits);
        if (nhits < 3)
          continue;
        if (tracks_view[it].quality() == Quality::edup)
          continue;
        // On hitContainer overflow, bulkFill returns kOverflow and the quality
        // stamp below is skipped, so the slot retains its pre-init value.  On a
        // zero-initialised SoA that is bad (0); on a GPU caching allocator it
        // can be garbage.  Skip such slots instead of asserting: the tuple was
        // already dropped (lossy truncation), so counting it here would be wrong.
        if (tracks_view[it].quality() != Quality::bad)
          continue;
        // On content-buffer overflow the offset is plugged (size is correct) but
        // the content is unwritten, so nhits can read garbage.  Clamp to the
        // physics maximum and skip: a tuple with > maxHitsOnTrack hits is an
        // overflow artifact, not a real track.
        if (nhits > TrackerTraits::maxHitsOnTrack)
          continue;
        auto const nsel = nSelectedHits<TrackerTraits>(foundNtuplets, it, hh);
        if (nsel < 3)
          continue;  // too few selected hits to fit
        tupleMultiplicity->count(acc, nsel);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillMultiplicity {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);

        if (nhits < 3)
          continue;
        if (tracks_view[it].quality() == Quality::edup)
          continue;
        // Skip overflow tuples (see Kernel_countMultiplicity for rationale).
        if (tracks_view[it].quality() != Quality::bad)
          continue;
        if (nhits > TrackerTraits::maxHitsOnTrack)
          continue;
        auto const nsel = nSelectedHits<TrackerTraits>(foundNtuplets, it, hh);
        if (nsel < 3)
          continue;
        tupleMultiplicity->fill(acc, nsel, it);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_classifyTracks {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsMultiView hh,
                                  QualityCuts<TrackerTraits> cuts,
                                  bool useTrackDNN,
                                  float trackDNNThreshold) const {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
      // Counters for diagnostic output
      uint32_t nTracks = 0;
      uint32_t nFitted = 0;
      uint32_t nNaN = 0;
      uint32_t nDoublets = 0;
      uint32_t nDuplicates = 0;
#endif

      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);
        if (nhits == 0)
          break;  // guard

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
        nTracks++;
#endif

        // if duplicate: not even fit
        if (tracks_view[it].quality() == Quality::edup) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nDuplicates++;
#endif
          continue;
        }

        // Skip overflow tuples (see Kernel_countMultiplicity for rationale).
        if (tracks_view[it].quality() != Quality::bad)
          continue;

        // mark doublets as bad
        if (nhits < 3) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nDoublets++;
#endif
          continue;
        }

        // if the fit has any invalid parameters, mark it as bad
        bool isNaN = false;
        for (int i = 0; i < 5; ++i) {
          isNaN |= edm::isNotFinite(tracks_view[it].state()(i));
        }
        // FIT-FAILURE RULE: a non-finite chi2 IS a failed fit, exactly like a non-finite parameter,
        // and no track whose fit failed may be promoted. The test must be explicit because every
        // promotion gate downstream is an FP comparison written in the REJECTING sense --
        // QualityCuts::strictCut returns `chi2 >= maxChi2`, the stub-curvature walk tests
        // `chi2Stub > cut` -- and a comparison with a NaN operand is false, so a NaN chi2 would PASS
        // them all. edm::isNotFinite is a bit-pattern test on the exponent field, so it keeps
        // working under -Ofast / -ffinite-math-only, where an `x != x` idiom would be folded away.
        isNaN |= edm::isNotFinite(tracks_view[it].chi2());
        // state(2) is the (finite) inverse pt: an exactly-zero value from a straight-line or
        // numerically-degenerate fit maps to an infinite momentum in the host local-to-global
        // transform, so treat it as bad here too and never promote such a track
        isNaN |= (tracks_view[it].state()(2) == 0.f);
        if (isNaN) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nNaN++;
          printf("FIT_DEBUG: Track %d has NaN - nhits=%d chi2=%f pt=%f eta=%f\n",
                 it,
                 nhits,
                 tracks_view[it].chi2(),
                 tracks_view[it].pt(),
                 tracks_view[it].eta());
#endif
          continue;
        }

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
        nFitted++;
        // Print details for first 10 successfully fitted tracks
        if (nFitted <= 10) {
          printf("FIT_DEBUG: Track %d FITTED - nhits=%d pt=%.3f eta=%.3f phi=%.3f chi2=%.3f tip=%.4f zip=%.4f\n",
                 it,
                 nhits,
                 tracks_view[it].pt(),
                 tracks_view[it].eta(),
                 tracks_view[it].state()(0),  // phi is state[0]
                 tracks_view[it].chi2(),
                 tracks_view[it].state()(1),   // tip is state[1]
                 tracks_view[it].state()(4));  // zip is state[4]
        }
#endif

        tracks_view[it].quality() = Quality::strict;

        bool failChi2 = cuts.strictCut(tracks_view, nhits, it);
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          const auto nHitsTot = hh.size();

          // ---- classify-embedded track classifier --------------------------------------
          // When enabled, the MLP score REPLACES the chi2-based strict->tight decision (both the
          // strictCut fit-chi2 gate AND the ntuplet-wide stub-consistency demotion below); the
          // fit chi2 and chi2Stub stay INPUTS of the network (caFitChi2, logChi2Stub). So once the DNN
          // decides a track we SKIP the stub-consistency walk entirely -- it only fed
          // maxNtupletStubChi2, whose verdict the score overwrites, so it would be wasted work. Feature
          // ORDER mirrors test/models/train_disp_nano.py FEATS (documented in CATrackDNNWeights.h).
          bool dnnHandled = false;
          if (useTrackDNN) {
            // Single-source feature fill (RecoTracker/PixelSeeding/interface/CATrackFeatures.h),
            // producing values identical to the host-side CA-features nano table producer's. On a
            // corrupt/short hit list fill() returns false -> fall through to the chi2-based path.
            caTrackFeatures::Features feat;
            static_assert(caTrackFeatures::kNFeat == caTrackDNN::kNFeat, "feature ABI mismatch");
            const bool featOk = caTrackFeatures::fill(foundNtuplets->begin(it),
                                                      foundNtuplets->end(it),
                                                      hh,
                                                      nHitsTot,
                                                      float(tracks_view[it].nLayers()),
                                                      tracks_view[it].chi2(),
                                                      feat,
                                                      /*extras=*/nullptr);
            // FIT-FAILURE RULE, gate half. This DNN gate REPLACED the classical `chi2 < maxChi2`
            // promotion, which rejected a failed fit as a side effect of NaN comparing false. The
            // network gives nothing for free: a non-finite input propagates through the MLP, and
            // the resulting score compared the wrong way round would promote the track. So the
            // finiteness of the network INPUTS is established BEFORE the network is evaluated --
            // never relying on a NaN surviving the sigmoid -- and a track with any non-finite
            // feature stays Quality::bad (quality() was optimistically set to strict above, so it
            // is written back explicitly). caFitChi2 is the fit chi2, already covered by the guard
            // at the top of the loop; this covers every other quantity the fill produced.
            const auto featArray = feat.asArray();
            bool featFinite = featOk;
            for (int k = 0; featFinite && k < int(caTrackFeatures::kNFeat); ++k)
              featFinite = !edm::isNotFinite(featArray[k]);
            if (featOk && !featFinite) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
              nNaN++;
#endif
              tracks_view[it].quality() = Quality::bad;
              continue;
            }
            if (featOk) {
              // Stage-1 high-recall loose->tight selector: a single threshold. The model retains
              // real/loose efficiency to large displacement; dedicated displaced fake rejection
              // belongs to a downstream selector.
              const float defThr = caTrackDNN::kDefaultThreshold;
              const float dnnThr = (trackDNNThreshold < 0.f) ? defThr : trackDNNThreshold;
              const float dnnScore = caTrackDNN_eval::score(featArray.data());
              // PROMOTING form on purpose: `score >= threshold` is the decision to PROMOTE and the
              // rejection is its negation, never `if (score < thr) reject`. Under -Ofast
              // (-ffinite-math-only) the compiler may assume no NaN operand and rewrite a rejecting
              // predicate into its finite-arithmetic complement, which would let a NaN score take
              // the promoting branch; in this form the default is "do not promote", so anything the
              // comparison cannot decide stays rejected.
              const bool dnnPromote = (dnnScore >= dnnThr);
              failChi2 = !dnnPromote;
              dnnHandled = true;
            }
          }

          // Ntuplet-wide stub-curvature consistency. A stub at transverse distance d0 from the beam line measures
          // kappa/2 + d0/r^2, so the per-stub curvatures of one track are linear in 1/r^2: that line is fitted by
          // weighted least squares (errors from the precision-only bend column) and the reduced chi2 of its residuals
          // (ndof = nStubs - 2) is the statistic; a chain admitted by the relaxed DCA still fails it and is demoted
          // below `tight`. Skipped when the DNN already decided.
#ifdef CA_CHI2_DUMP
          const bool computeStubChi2 = true;
#else
          const bool computeStubChi2 = !dnnHandled;
#endif
          if (computeStubChi2) {
            int nStubK = 0;
            float sumW = 0.f, sumWX = 0.f, sumWXX = 0.f, sumWK = 0.f, sumWKX = 0.f, sumWKK = 0.f;
            for (auto h = foundNtuplets->begin(it); h != foundNtuplets->end(it); ++h) {
              if (*h >= static_cast<unsigned int>(nHitsTot))
                break;  // content buffer corruption from overflow
              if (!isStub(hh, *h))
                continue;  // pixel hit
              auto const stub = hh.stub(int32_t(*h));
              const float s = stub.dPhiDrErrorPrec();
              if (s > 0.f) {
                const float d = stub.dPhiDr();
                const float xg = hh[*h].xGlobal();
                const float yg = hh[*h].yGlobal();
                const float rg2 = xg * xg + yg * yg;
                if (!(rg2 > 0.f))
                  continue;
                float den, w;  // same shared kappa formula as CATrackFeatures::fill
                caTrackFeatures::stubDenWeight(rg2, d, s, den, w);
                const float k = d / std::sqrt(den);  // stub curvature
                const float x = 1.f / rg2;           // the d0 term enters linearly in 1/r^2
                // hit precision plus multiple scattering, as in the doublet and triplet cuts
                const float sMS = caStubMS::kThetaPerCurv * 2.f * std::abs(k) / std::sqrt(rg2);
                w = 1.f / (1.f / w + sMS * sMS);
                sumW += w;
                sumWX += w * x;
                sumWXX += w * x * x;
                sumWK += w * k;
                sumWKX += w * k * x;
                sumWKK += w * k * k;
                ++nStubK;
              }
            }
            // chi2Stub < 0 => not enough stubs to judge consistency.
            float chi2Stub = -1.f;
            if (nStubK >= 3 && sumW > 0.f) {
              const float det = sumW * sumWXX - sumWX * sumWX;
              if (std::abs(det) > 0.f) {
                // k = a + b/r^2, a = kappa/2 and b = d0
                const float a = (sumWXX * sumWK - sumWX * sumWKX) / det;
                const float b = (sumW * sumWKX - sumWX * sumWK) / det;
                chi2Stub = (sumWKK - a * sumWK - b * sumWKX) / float(nStubK - 2);
              }
              if (chi2Stub >= 0.f && !dnnHandled && cuts.maxNtupletStubChi2 >= 0.f) {
                // The keep decision is the positive comparison (chi2Stub <= cut), so a non-finite chi2Stub
                // from a degenerate stub set falls to the demoting side; the explicit isNotFinite keeps that
                // true under -Ofast.
                const bool stubConsistent = !edm::isNotFinite(chi2Stub) && (chi2Stub <= cuts.maxNtupletStubChi2);
                if (!stubConsistent)
                  failChi2 = true;
              }
            }
#ifdef CA_CHI2_DUMP
            // Per-track calibration dump: fit chi2 vs ntuplet-wide stub consistency.
            // On a pure-signal run every dumped track is real; on displaced+PU it shows
            // the real/fake mix. Define CA_CHI2_DUMP and run a few events.
            printf("[Chi2Dump] nhits=%d nStubK=%d chi2=%.4f chi2Stub=%.4f pt=%.4f eta=%.4f\n",
                   nhits,
                   nStubK,
                   tracks_view[it].chi2(),
                   chi2Stub,
                   tracks_view[it].pt(),
                   tracks_view[it].eta());
#endif
          }
        }
        if (failChi2)
          continue;

        tracks_view[it].quality() = Quality::tight;

        if (cuts.isHP(tracks_view, nhits, it))
          tracks_view[it].quality() = Quality::highPurity;
      }

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("FIT_DEBUG SUMMARY: total=%d fitted=%d NaN=%d doublets=%d duplicates=%d\n",
               nTracks,
               nFitted,
               nNaN,
               nDoublets,
               nDuplicates);
      }
#endif
    }
  };

  template <typename TrackerTraits>
  class Kernel_doStatsForTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  Counters *counters) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  //guard
        if (tracks_view[idx].quality() < Quality::loose)
          continue;
        alpaka::atomicAdd(acc, &(counters->nLooseTracks), 1ull, alpaka::hierarchy::Blocks{});
        if (tracks_view[idx].quality() < Quality::strict)
          continue;
        alpaka::atomicAdd(acc, &(counters->nGoodTracks), 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Final quality distribution counter: counts tracks at each quality level
  // after ALL processing (classification, fishbone, duplicate removal).
#ifdef CA_PIPELINE_COUNTERS
  // Runs right before the pipeline printout to complete the diagnostic funnel.
  template <typename TrackerTraits>
  class Kernel_countFinalQuality {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsMultiView hh,
                                  uint32_t *__restrict__ pipelineCounters) const {
      using Quality = pixelTrack::Quality;
      using PC = caHitNtupletGenerator::PipelineCounter;

      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(idx);
        if (nhits == 0)
          break;  // guard

        alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTotal], 1u, alpaka::hierarchy::Blocks{});

        auto q = tracks_view[idx].quality();
        if (q == Quality::bad) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualBad], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::edup) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualEdup], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::dup) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualDup], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::loose) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualLoose], 1u, alpaka::hierarchy::Blocks{});
        } else {
          // strict, tight, or highPurity -- check OT once for all levels
          bool hasOT = false;
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            auto nHits = hh.size();
            for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
              if (*h >= static_cast<unsigned int>(nHits))
                break;  // content buffer corruption from overflow
              if (isStub(hh, *h)) {
                hasOT = true;
                break;
              }
            }
          }
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict], 1u, alpaka::hierarchy::Blocks{});
          if (hasOT)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrictWithOT], 1u, alpaka::hierarchy::Blocks{});
          if (q >= Quality::tight) {
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight], 1u, alpaka::hierarchy::Blocks{});
            if (hasOT)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTightWithOT], 1u, alpaka::hierarchy::Blocks{});
          }
          if (q >= Quality::highPurity) {
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP], 1u, alpaka::hierarchy::Blocks{});
            if (hasOT)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHPWithOT], 1u, alpaka::hierarchy::Blocks{});
          }

          // Per-nhits quality breakdown
          float chi2 = tracks_view[idx].chi2();
          if (nhits <= 4) {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict34], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight34], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP34], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 0.9f && chi2 < 1.1f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary34], 1u, alpaka::hierarchy::Blocks{});
          } else if (nhits == 5) {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict5], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight5], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP5], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 2.7f && chi2 < 3.3f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary5], 1u, alpaka::hierarchy::Blocks{});
          } else {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict6p], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight6p], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP6p], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 4.5f && chi2 < 5.5f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary6p], 1u, alpaka::hierarchy::Blocks{});
          }

          // Fishbone-hit multiplicity per track. The hit container does not mark fishbone hits and
          // the cell count is not available here, so every track lands in the 0-fishbone bucket.
          uint32_t nFishbone = 0;
          nFishbone = 0;
          if (nFishbone == 0)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone0], 1u, alpaka::hierarchy::Blocks{});
          else if (nFishbone == 1)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone1], 1u, alpaka::hierarchy::Blocks{});
          else
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone2p], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };
#endif  // CA_PIPELINE_COUNTERS

  template <typename TrackerTraits>
  class Kernel_countHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple *hitToTuple,
                                  uint32_t nHits) const {  // OT extras bin at nHits + otIdx
      const auto nKeys = hitToTuple->nOnes();
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  // guard
        for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
          auto const key = *h;
          // Key-range guard: a hitContainer content overflow leaves unwritten (garbage) hit ids in
          // the CSR, so the key can land outside [0, nOnes). Drop instead of writing outside off[].
          // The drop is counted once, on the fill pass below, which skips exactly the same keys.
          if (key < nKeys)
            hitToTuple->count(acc, key);
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple *hitToTuple,
                                  uint32_t nHits,
                                  Counters *counters) const {  // OT extras bin at nHits + otIdx
      const auto nKeys = hitToTuple->nOnes();
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  // guard
        for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
          auto const key = *h;
          // Key-range guard, mirroring the count pass; the drop is counted here, once per lost
          // hit->tuple association.
          if (key < nKeys)
            hitToTuple->fill(acc, key, idx);
          else
            alpaka::atomicAdd(acc, &counters->nHitToTupleOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Content-buffer overflow repair, paired with the truncating bulkFill in OneToManyAssoc.h: when a tuple's hit
  // block does not fit the container, bulkFill plugs the offset but writes no content. Run right after
  // bulkFinalize, these kernels clamp every offset to the start of the first overflowed tuple k, so that tuple
  // and all later ones become empty. No-op when nothing overflowed. clampInfo[0] = off[k], clampInfo[1] = k,
  // which Kernel_fillHitDetIndices uses to cut nTracks so that no empty slot is published.
  class Kernel_findTupleContentOverflow {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  uint32_t *__restrict__ clampInfo) const {
      const uint32_t cap = uint32_t(foundNtuplets->capacity());
      const uint32_t nOff = uint32_t(foundNtuplets->totOnes());  // off[] has nOnes()+1 entries
      for (auto k : cms::alpakatools::uniform_elements(acc, nOff - 1)) {
        if (foundNtuplets->off[k] <= cap && foundNtuplets->off[k + 1] > cap) {
          clampInfo[0] = foundNtuplets->off[k];
          clampInfo[1] = uint32_t(k);
        }
      }
    }
  };

  class Kernel_clampTupleContentOverflow {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitContainer *__restrict__ foundNtuplets,
                                  uint32_t const *__restrict__ clampInfo) const {
      const uint32_t bound = clampInfo[0];
      for (auto j : cms::alpakatools::uniform_elements(acc, uint32_t(foundNtuplets->totOnes()))) {
        if (foundNtuplets->off[j] > bound)
          foundNtuplets->off[j] = bound;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitDetIndices {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  TkHitSoAView track_hits_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsMultiView hh,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  uint32_t const *__restrict__ tupleClampInfo) const {
      // clamp the number of tracks to the capacity of the SoA
      auto ntracks = std::min<int>(apc->get().first, tracks_view.metadata().size() - 1);
      // ... and to the first tuple dropped by a content-buffer overflow: the slots from that tuple on are
      // empty after the repair and must not be published (0xFFFFFFFF = nothing overflowed, no cut).
      if (tupleClampInfo[1] < uint32_t(ntracks))
        ntracks = int(tupleClampInfo[1]);
      if (cms::alpakatools::once_per_grid(acc))
        tracks_view.nTracks() = ntracks;

      // copy offsets, clamped to the hit SoA capacity: on a content-buffer overflow the raw offset
      // can exceed what the copy loop below writes, and a CSR end past the copied region would make
      // downstream hit walks read unwritten rows. The clamp keeps the CSR self-consistent with the
      // truncated copy (offset for track 0 is always 0).
      const uint32_t hitRowCap = uint32_t(track_hits_view.metadata().size());
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        tracks_view[idx].hitOffsets() = std::min(foundNtuplets->off[idx + 1], hitRowCap);
        tracks_view[idx].ndof() = 0;  // stamped by the fit for fitted tuples
      }
      // Tail: the slots past the last track carry its CSR end offset, so nHits() reads zero there and a
      // reader that walks the SoA up to the first empty slot stops at the right place.
      const uint32_t hitEndTail = std::min(foundNtuplets->off[ntracks], hitRowCap);
      for (auto idx : cms::alpakatools::uniform_elements(acc, uint32_t(tracks_view.metadata().size())))
        if (int(idx) >= ntracks)
          tracks_view[idx].hitOffsets() = hitEndTail;
      // fill hit indices, clamped to the hit SoA capacity: foundNtuplets->size() is the
      // AtomicPairCounter's hits-in-tracks total, which on a tuple overflow exceeds what was actually
      // written, so an unclamped loop would read the container beyond its filled region. The clamp
      // never binds while the tuple cap is not reached; it is here so that an overflow degrades
      // rather than corrupts.
      const uint32_t nHitsInTracks = std::min<uint32_t>(foundNtuplets->size(), track_hits_view.metadata().size());
      for (auto idx : cms::alpakatools::uniform_elements(acc, nHitsInTracks)) {
        // On content-buffer overflow the content is unwritten (garbage), so the
        // hit index can be out of range.  Skip such entries: the hit was already
        // dropped (lossy truncation), so writing a garbage detId would corrupt.
        if (foundNtuplets->content[idx] >= (uint32_t)hh.size())
          continue;
        track_hits_view[idx].id() = foundNtuplets->content[idx];
        track_hits_view[idx].detId() = hh[foundNtuplets->content[idx]].detectorIndex();
        track_hits_view[idx].attached() = 0;  // hit found by the CA
#ifdef CA_DEBUG
        printf("Kernel_fillHitDetIndices %d %d %d \n",
               idx,
               foundNtuplets->content[idx],
               track_hits_view.metadata().size());
#endif
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_doStatsForHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitToTuple const *__restrict__ hitToTuple,
                                  Counters *counters) const {
      auto &c = *counters;
      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple->nOnes())) {
        if (hitToTuple->size(idx) == 0)
          continue;  // SHALL NOT BE break
        alpaka::atomicAdd(acc, &c.nUsedHits, 1ull, alpaka::hierarchy::Blocks{});
        if (hitToTuple->size(idx) > 1)
          alpaka::atomicAdd(acc, &c.nDupHits, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_countSharedHit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  int *__restrict__ nshared,
                                  HitContainer const *__restrict__ ptuples,
                                  Quality const *__restrict__ quality,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      constexpr auto loose = Quality::loose;

      auto &hitToTuple = *phitToTuple;
      auto const &foundNtuplets = *ptuples;
      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        int nt = 0;

        // count "good" tracks
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (quality[*it] < loose)
            continue;
          ++nt;
        }

        if (nt < 2)
          continue;

        // now mark  each track triplet as sharing a hit
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (foundNtuplets.size(*it) > 3)
            continue;
          alpaka::atomicAdd(acc, &nshared[*it], 1, alpaka::hierarchy::Blocks{});
        }

      }  //  hit loop
    }
  };

  template <typename TrackerTraits>
  class Kernel_markSharedHit {
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  int const *__restrict__ nshared,
                                  HitContainer const *__restrict__ tuples,
                                  Quality *__restrict__ quality,
                                  bool dupPassThrough) const {
      // constexpr auto bad = Quality::bad;
      constexpr auto dup = Quality::dup;
      constexpr auto loose = Quality::loose;
      // constexpr auto strict = Quality::strict;

      // quality to mark rejected
      auto const reject = dupPassThrough ? loose : dup;
      for (auto idx : cms::alpakatools::uniform_elements(acc, tuples->nOnes())) {
        if (tuples->size(idx) == 0)
          break;  //guard
        if (quality[idx] <= reject)
          continue;
        if (nshared[idx] > 2)
          quality[idx] = reject;
      }
    }
  };

  // Track-parallel single-writer shared-hit removers: each thread owns one track, inspects the hit
  // buckets it belongs to (hitToTuple), reads every quality from the frozen scratch snapshot, and writes
  // only its own track's quality()
  template <typename TrackerTraits>
  class Kernel_rejectDuplicate {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      auto score = [&](uint32_t it) { return tracks_view[it].chi2(); };

      // A track is rejected iff some compatible track sharing one of its hits is strictly better by
      // the total order (more layers, then higher quality, then lower chi2, then lower track index)
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        auto const qi = qual(it);
        if (qi <= reject)
          continue;
        auto const nli = tracks_view[it].nLayers();
        // Phase2OTStubs only: the duplicate winner ordering inserts the TOTAL HIT COUNT as a tie-break
        // BETWEEN nLayers and quality, giving
        //   winner = max nLayers -> max total hits -> max quality -> min chi2 -> min index.
        // reco::nHits() is the track's full CSR hit extent, so it favours the hit-richer member
        // without special-casing subdetectors and separates tracks that nLayers alone ties (forward
        // duplicates tying on nLayers would otherwise fall straight to the chi2 tie-break, letting a
        // pixel-rich prompt track beat its OT-rich displaced twin and losing its TID hits). Off on
        // every other topology, where the ordering is the upstream one.
        constexpr bool kUseHitCountTieBreak = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
        const uint32_t nhi = kUseHitCountTieBreak ? ::reco::nHits(tracks_view, it) : 0u;

        // get track parameters and covariances
        float iParams[nTrackParameters];
        float iCovs[nTrackParameters];
        for (int p{0}; p < nTrackParameters; ++p) {
          iParams[p] = tracks_view[it].state()(p);
          iCovs[p] = tracks_view[it].covariance()(iParam2iCov[p]);
        }
        auto incompatibleTrackParams = [&](uint32_t jt) -> bool {
          for (int p{0}; p < nTrackParameters; ++p) {
            const auto dpij = iParams[p] - tracks_view[jt].state()(p);
            const auto e2dpij = fastDupNSigma2 * (iCovs[p] + tracks_view[jt].covariance()(iParam2iCov[p]));
            if (dpij * dpij > e2dpij)
              return true;
          }
          return false;
        };

        bool dominated = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !dominated; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (jt == it)
              continue;
            auto const qj = qual(jt);
            if (qj <= reject)
              continue;
            if (incompatibleTrackParams(jt))
              continue;
            auto const nlj = tracks_view[jt].nLayers();
            // jt dominates it by the total order (nLayers, [total hits], quality, score, then track
            // index). The score test stays a strict order even for a non-finite score (NaN), so
            // exactly one of a duplicate pair is always demoted
            bool jBetterTail =
                (qj > qi || (qj == qi && (score(jt) < score(it) || (!(score(it) < score(jt)) && jt < it))));
            bool jBetter;
            if constexpr (kUseHitCountTieBreak) {
              const uint32_t nhj = ::reco::nHits(tracks_view, jt);
              jBetter = (nlj > nli) || (nlj == nli && (nhj > nhi || (nhj == nhi && jBetterTail)));
            } else {
              jBetter = (nlj > nli) || (nlj == nli && jBetterTail);
            }
            if (jBetter) {
              dominated = true;
              break;
            }
          }
        }
        if (dominated)
          tracks_view[it].quality() = reject;
      }
    }
  };

  // Phase-1 specialization (very forward triplets)
  template <>
  class Kernel_rejectDuplicate<pixelTopology::Phase1> {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      auto score = [&](uint32_t it) { return std::abs(reco::tip(tracks_view, it)); };

      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        auto const qi = qual(it);
        if (qi <= reject)
          continue;
        auto const opi = tracks_view[it].state()(2);
        auto const e2opi = tracks_view[it].covariance()(9);
        auto const cti = tracks_view[it].state()(3);
        auto const e2cti = tracks_view[it].covariance()(12);
        auto const nli = tracks_view[it].nLayers();

        bool dominated = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !dominated; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (jt == it)
              continue;
            auto const qj = qual(jt);
            if (qj <= reject)
              continue;
            auto const opj = tracks_view[jt].state()(2);
            auto const ctj = tracks_view[jt].state()(3);
            auto const dct = nSigma2Phase1 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto const dop = nSigma2Phase1 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            auto const nlj = tracks_view[jt].nLayers();
            // jt dominates it by the total order (nLayers, quality, score, then track index). The score
            // test stays a strict order even for a non-finite score (NaN), so exactly one of a duplicate
            // pair is always demoted
            bool jBetter =
                (nlj > nli) ||
                (nlj == nli &&
                 (qj > qi || (qj == qi && (score(jt) < score(it) || (!(score(it) < score(jt)) && jt < it)))));
            if (jBetter) {
              dominated = true;
              break;
            }
          }
        }
        if (dominated)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_sharedHitCleaner {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  uint32_t const *__restrict__ layerStarts,
                                  TkSoAView tracks_view,
                                  int nmin,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      // quality of longest track
      auto const longTqual = Quality::highPurity;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      uint32_t l1end = layerStarts[1];

      // Short track `it` (nLayers <= nmin) is killed if it shares a non-bpix1 hit with a longer track
      // (nLayers == maxNl >= 4 among the highPurity tracks of that hit). maxNl is a reduction over the
      // frozen snapshot, so this is order-independent
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject)
          continue;
        auto const nlit = tracks_view[it].nLayers();
        if (nlit > nmin)
          continue;  // only short tracks are cleaned here

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h < l1end)
            continue;  // shared hit on bpix1
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          int8_t maxNl = 0;
          if (hitToTuple.size(h) >= 2) {
            for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
              if (qual(*jp) < longTqual)
                continue;
              maxNl = std::max(tracks_view[*jp].nLayers(), maxNl);
            }
          }

          // For Phase2OTStubs: several stubs sharing a lowerHitIdx have different hit indices but stand for
          // the same physical measurement, so for cleaning purposes they count as the same shared hit.
          // PS only: a 2S stub now carries its lower cluster id as well, but merging 2S stubs here kills
          // short tracks whose long partner does not replace them -- measured, 4.5 points of prompt barrel
          // efficiency on ttbar PU200. The rule stays where it was tuned until it is retuned.
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            if (h < static_cast<uint32_t>(hh.size()) && isStub(hh, h) && hh.stub(int32_t(h)).isPS()) {
              auto const lowerHitIdx = hh.stub(int32_t(h)).lowerHitIdx();
              if (lowerHitIdx != std::numeric_limits<uint32_t>::max()) {
                auto const offsetStubs = caStructures::offsetStubsOf(hh);
                auto const nHits = static_cast<uint32_t>(hh.size());
                for (uint32_t otherIdx = offsetStubs; otherIdx < nHits; ++otherIdx) {
                  if (otherIdx == h)
                    continue;
                  if (otherIdx >= hitToTuple.nOnes())
                    continue;
                  if (!isStub(hh, otherIdx) || !hh.stub(int32_t(otherIdx)).isPS())
                    continue;
                  if (hh.stub(int32_t(otherIdx)).lowerHitIdx() != lowerHitIdx)
                    continue;
                  for (auto jp = hitToTuple.begin(otherIdx); jp != hitToTuple.end(otherIdx); ++jp) {
                    if (qual(*jp) < longTqual)
                      continue;
                    maxNl = std::max(tracks_view[*jp].nLayers(), maxNl);
                  }
                }
              }
            }
          }

          if (maxNl >= 4 && nlit < maxNl)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_tripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::strict;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };

      // Track `it` is rejected if, on one of its shared hits whose good-quality tracks are all
      // triplets, it is not the best-tip survivor (lower track index breaks ties)
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject)
          continue;

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          if (hitToTuple.size(h) < 2)
            continue;
          bool onlyTriplets = true;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            if (qual(*jp) <= good)
              continue;
            onlyTriplets &= reco::isTriplet(tracks_view, *jp);
            if (!onlyTriplets)
              break;
          }
          if (!onlyTriplets)
            continue;
          float mc = maxScore;
          uint32_t im = tkNotFound;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (qual(jt) >= good) {
              auto const t = std::abs(reco::tip(tracks_view, jt));
              if (t < mc || (t == mc && jt < im)) {
                mc = t;
                im = jt;
              }
            }
          }
          if (im != tkNotFound && it != im)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_simpleTripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::loose;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };

      // Triplet `it` is rejected if, on one of its shared hits, it is not the best-tip survivor
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject || !reco::isTriplet(tracks_view, it))
          continue;

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          if (hitToTuple.size(h) < 2)
            continue;
          float mc = maxScore;
          uint32_t im = tkNotFound;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (qual(jt) >= good) {
              auto const t = std::abs(reco::tip(tracks_view, jt));
              if (t < mc || (t == mc && jt < im)) {
                mc = t;
                im = jt;
              }
            }
          }
          if (im != tkNotFound && it != im)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_print_found_ntuplets {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  uint32_t firstPrint,
                                  uint32_t lastPrint,
                                  int iev) const {
      constexpr auto loose = Quality::loose;

      for (auto i : cms::alpakatools::uniform_elements(acc, firstPrint, std::min(lastPrint, foundNtuplets->nOnes()))) {
        auto nh = foundNtuplets->size(i);
        if (nh < 3)
          continue;
        if (tracks_view[i].quality() < loose)
          continue;
        printf("TK: %d %d %d %d %f %f %f %f %f %f %f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
               10000 * iev + i,
               int(tracks_view[i].quality()),
               nh,
               tracks_view[i].nLayers(),
               reco::charge(tracks_view, i),
               tracks_view[i].pt(),
               tracks_view[i].eta(),
               reco::phi(tracks_view, i),
               reco::tip(tracks_view, i),
               reco::zip(tracks_view, i),
               tracks_view[i].chi2(),
               hh[*foundNtuplets->begin(i)].zGlobal(),
               hh[*(foundNtuplets->begin(i) + 1)].zGlobal(),
               hh[*(foundNtuplets->begin(i) + 2)].zGlobal(),
               nh > 3 ? hh[int(*(foundNtuplets->begin(i) + 3))].zGlobal() : 0,
               nh > 4 ? hh[int(*(foundNtuplets->begin(i) + 4))].zGlobal() : 0,
               nh > 5 ? hh[int(*(foundNtuplets->begin(i) + 5))].zGlobal() : 0,
               nh > 6 ? hh[int(*(foundNtuplets->begin(i) + nh - 1))].zGlobal() : 0);
      }
    }
  };

  class Kernel_printCounters {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc, Counters const *counters) const {
      auto const &c = *counters;
      printf(
          "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nLooseTracks  |  nGoodTracks | nUsedHits | "
          "nDupHits | nFishCells | nKilledCells | nUsedCells | nZeroTrackCells ||\n");
      printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
             c.nEvents,
             c.nHits,
             c.nCells,
             c.nTuples,
             c.nFitTracks,
             c.nLooseTracks,
             c.nGoodTracks,
             c.nUsedHits,
             c.nDupHits,
             c.nFishCells,
             c.nKilledCells,
             c.nEmptyCells,
             c.nZeroTrackCells);
      printf(
          "Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f|  %.3f|  "
          "%.3f||\n",
          c.nEvents,
          c.nHits / double(c.nEvents),
          c.nCells / double(c.nEvents),
          c.nTuples / double(c.nEvents),
          c.nFitTracks / double(c.nEvents),
          c.nLooseTracks / double(c.nEvents),
          c.nGoodTracks / double(c.nEvents),
          c.nUsedHits / double(c.nEvents),
          c.nDupHits / double(c.nEvents),
          c.nFishCells / double(c.nCells),
          c.nKilledCells / double(c.nCells),
          c.nEmptyCells / double(c.nCells),
          c.nZeroTrackCells / double(c.nCells));
      printf(
          "Counters Overflow %lld ||  tupleOvf=%lld  cellOvf=%lld  tripletOvf=%lld  "
          "cellTrackOvf=%lld  hitToTupleOvf=%lld  hitToCellOvf=%lld ||\n",
          c.nEvents,
          c.nTupleOverflow,
          c.nCellOverflow,
          c.nTripletOverflow,
          c.nCellTrackOverflow,
          c.nHitToTupleOverflow,
          c.nHitToCellOverflow);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels

// The host side of CAHitNtupletGeneratorKernels: the launchers and the buffer management. Instantiated per
// topology by the CAHitNtupletGeneratorKernels_<Topology>.dev.cc translation units.
// C++ headers
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"  // finalDedup count-and-clamp overflow warning
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// local headers
#include "CAFishbone.h"
#include "CAHitNtupletGeneratorKernels.h"

//#define GPU_DEBUG
// #define NTUPLE_DEBUG
//#define CA_STATS

namespace ALPAKA_ACCELERATOR_NAMESPACE {

// The sizing accumulator is used by the CA_STATS report and by the GPU_DEBUG
// per-event allocation report, so it must be compiled for either toggle.
#if defined(CA_STATS) || defined(GPU_DEBUG)
  // Define a thread and event safe accumulator for sizing-parameter
  // recommendations based on observed maxima and averages.
  //
  // Fixed-avg parameters (size = max_keys * avg):
  //   index 0 -> avgHitsPerTrack
  //   index 1 -> avgCellsPerHit
  //   index 2 -> avgCellsPerCell
  //   index 3 -> avgTracksPerCell
  //
  // Scaling-with-nHits parameters (size = slope * nHits):
  //   index 0 -> maxNumberOfDoublets (slope = nCells  / nHits)
  //   index 1 -> maxNumberOfTuples   (slope = nTuples / nHits)
  // Track maxNHits as well so f(nHits) can be evaluated at the
  // observed worst case for a concrete recommendation.
  namespace {
    struct SizingAccumulator {
      std::mutex mtx;
      uint64_t n = 0;
      double maxReq[4] = {0., 0., 0., 0.};
      double sumReq[4] = {0., 0., 0., 0.};
      double maxPhy[4] = {0., 0., 0., 0.};
      double sumPhy[4] = {0., 0., 0., 0.};
      double maxRatio[2] = {0., 0.};
      double sumRatio[2] = {0., 0.};
      uint32_t maxNHits = 0u;
      // Aggregate allocation tracking
      uint64_t allocN = 0;
      uint64_t allocBytesSum = 0;
      uint64_t allocBytesMax = 0;
    };
    // Single accumulator: the recommendations are aggregated over every CA instance
    // running in the job.
    inline SizingAccumulator &sizingAccumulator() {
      static SizingAccumulator s;
      return s;
    }
  }  // namespace
#endif  // CA_STATS || GPU_DEBUG

  // Sizing rule of the cells + cell->track-offsets arena (see the member declaration in the header).
  // Returns the arena's extent in SimpleCell units, or 0 when one allocation is not cheaper than two.
  template <typename TrackerTraits>
  uint32_t CAHitNtupletGeneratorKernels<TrackerTraits>::cellArenaExtent(uint32_t cellBound, uint32_t maxDoublets) {
    // The allocator's bin: the smallest power of two >= bytes, floored at its 256 B minimum bin
    // (binGrowth 2, minBin 8, maxBin 30 -- HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h).
    auto bin = [](std::size_t bytes) {
      std::size_t b = 256;
      while (b < bytes)
        b *= 2;
      return b;
    };
    constexpr std::size_t kCell = sizeof(SimpleCell);
    // 64 cells = 64*12 = 768 B, a multiple of 256, so the offsets region starts 256 B aligned
    // whatever the cap is.
    const std::size_t cells = ((std::size_t(cellBound) + 63u) / 64u) * 64u;
    const std::size_t offsetBytes = (std::size_t(maxDoublets) + 1u) * sizeof(GenericContainerOffsets);
    const std::size_t extra = (offsetBytes + kCell - 1u) / kCell;
    const std::size_t separate = bin(std::size_t(cellBound) * kCell) + bin(offsetBytes);
    const std::size_t together = bin((cells + extra) * kCell);
    return (together < separate) ? uint32_t(cells + extra) : 0u;
  }

  // Which of the three packings of the four cell-scale buffers is cheapest for this event's bounds.
  // The allocator's bins are powers of two, so which packing wins depends on where 12*cellBound falls
  // inside its bin; all three are evaluated and the smallest total is taken.
  // The minimum is over the bytes ALLOCATED for the four buffers, not the bytes resident at every
  // instant of the event: kThreeCellIndexArrays puts the hit->cell storage and the cell->cell offsets
  // in the same buffer as the cell->track offsets, which lives to the end of the event, so
  // releaseBuildScratch() cannot hand those two back after launchKernels. It still allocates less in
  // the largest events and lowers the per-stream peak live set, which is what a smaller device or more
  // streams per job would feel first.
  template <typename TrackerTraits>
  typename CAHitNtupletGeneratorKernels<TrackerTraits>::CellLayout
  CAHitNtupletGeneratorKernels<TrackerTraits>::chooseCellLayout(uint32_t cellBound, uint32_t maxDoublets) {
    auto bin = [](std::size_t bytes) {
      std::size_t b = 256;
      while (b < bytes)
        b *= 2;
      return b;
    };
    auto pad = [](std::size_t b) { return (b + 255u) / 256u * 256u; };
    const std::size_t cells = std::size_t(cellBound) * sizeof(SimpleCell);
    const std::size_t store = std::size_t(std::max(cellBound, 1u)) * sizeof(GenericContainerStorage);
    const std::size_t off = (std::size_t(maxDoublets) + 1u) * sizeof(GenericContainerOffsets);
    const std::size_t sep = bin(cells) + bin(store) + 2u * bin(off);
    const std::size_t a = bin(pad(cells) + off) + bin(store) + bin(off);
    const std::size_t b3 = bin(cells) + bin(pad(store) + pad(off) + off);
    if (b3 < sep && b3 <= a)
      return CellLayout::kThreeCellIndexArrays;
    if (a < sep)
      return CellLayout::kCellsWithTrackOffsets;
    return CellLayout::kSeparate;
  }

  template <typename TrackerTraits>
  CAHitNtupletGeneratorKernels<TrackerTraits>::CAHitNtupletGeneratorKernels(Params const &params,
                                                                            uint32_t nHits,
                                                                            uint32_t offsetBPIX2,
                                                                            uint32_t maxDoublets,
                                                                            uint32_t maxTuples,
                                                                            uint16_t nLayers,
                                                                            Queue &queue)
      : m_params(params) {
    //////////////////////////////////////////////////////////
    // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
    //////////////////////////////////////////////////////////

    counters_ = cms::alpakatools::make_device_buffer<Counters>(queue);
    // Here we define the OneToMany maps and the histograms
    // allocating the buffers and defining the views.
    // For each map/histo, we need:
    // - a buffer for the offsets sized as the number of ones + 1
    //   (with the last bin holding the total number of ones)
    // - a buffer fot the content/storage itself sized as the number of many

    auto const &algoParams = m_params.algoParams_;
    // Allocation strategy (see fillDescriptions). When both delayAllocations_ and countDoubletsFirst_
    // are false every buffer is allocated here with no device->host synchronization
    // Defer cell-derived + hit->track buffers
    const bool delay = algoParams.delayAllocations_;
    // defer simpleCells/hitToCellStorage and allocate to the actual number of doublets produced
    const bool countFirst = algoParams.countDoubletsFirst_;

    // CELL BOUND: bounds the cell array and the hit->cell association (one entry per cell).
    // The bin-edge cap is a stubs-chain sizing choice; the default chain keeps the configured maxDoublets.
    static constexpr uint32_t kMaxDoubletsForCellBin = (64u * 1024u * 1024u) / uint32_t(sizeof(SimpleCell));
    uint32_t cellBound = maxDoublets;
    if constexpr (std::is_same_v<TrackerTraits, pixelTopology::Phase2OTStubs>)
      cellBound = std::min(maxDoublets, kMaxDoubletsForCellBin);
    // Packing of the four cell-scale buffers, evaluated per event (see chooseCellLayout). Only the
    // up-front allocation mode packs: the exact-allocation paths size these buffers from counts that do
    // not exist yet at this point.
    const CellLayout cellLayout =
        (!delay && !countFirst) ? chooseCellLayout(cellBound, maxDoublets) : CellLayout::kSeparate;
    if (cellLayout == CellLayout::kThreeCellIndexArrays) {
      // hit->cell storage, then the two cell-keyed offsets, each 256 B aligned inside one buffer.
      const std::size_t nStore = ((std::size_t(std::max(cellBound, 1u)) + 63u) / 64u) * 64u;
      const std::size_t nOff = ((std::size_t(maxDoublets) + 1u + 63u) / 64u) * 64u;
      device_cellIndexArena_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(
          queue, uint32_t(nStore + nOff + std::size_t(maxDoublets) + 1u));
      arenaHitToCellStorage_ = device_cellIndexArena_->data();
      arenaCellToNeighborsOffsets_ =
          reinterpret_cast<GenericContainerOffsets *>(device_cellIndexArena_->data() + nStore);
      arenaCellToTracksOffsets_ =
          reinterpret_cast<GenericContainerOffsets *>(device_cellIndexArena_->data() + nStore + nOff);
    }
    uint32_t outerHits =
        nHits - offsetBPIX2;  // the number of hits that may be used as outer hits for a cell (so not on bpix1)

    // Sizes the per-track hit storage (hitContainer/hitToTuple). Same expression as the output
    // trackHits SoA in CAHitNtupletGenerator::beginTuplesAsync, so the two capacities match.
    uint32_t nHitsToTracks = caHitNtupletGenerator::nHitRowsForTuples(maxTuples, algoParams.avgHitsPerTrack_);

#ifdef GPU_DEBUG
    std::cout << "Allocation for tuple building with: " << std::endl;
    std::cout << "- nHits          = " << nHits << std::endl;
    std::cout << "- outerHits      = " << outerHits << std::endl;
    std::cout << "- maxDoublets    = " << maxDoublets << std::endl;
    std::cout << "- maxTracks      = " << maxTuples << std::endl;
    std::cout << "- nHitsToTracks  = " << nHitsToTracks << std::endl;
#endif

    // Hits -> Track
    // The handle and the per-hit offsets (key space) are always allocated here. The
    // storage holds one entry per hit-in-track: with delayAllocations_ it is sized from
    // the actual hits-in-tracks count in allocateAfterNtuplets (after launchKernels)
    // Otherwise it is allocated here at the nHitsToTracks safety cap
    device_hitToTuple_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToTupleOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nHits + 1);
    if (!delay) {
      device_hitToTupleStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToTracks);
      device_hitToTupleView_ = {device_hitToTuple_->data(),
                                device_hitToTupleOffsets_->data(),
                                device_hitToTupleStorage_->data(),
                                nHits + 1,
                                nHitsToTracks};

      HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);
    }

    // (Outer) Hits-> Cells
    // The storage holds exactly one entry per cell
    // With countDoubletsFirst it is sized to the exact nCells in buildDoublets
    // Otherwise to the maxDoublets safety cap here -- the same cap that bounds the cell array
    // itself (device_simpleCells_) and that the doublet finder enforces when it appends. Sizing
    // this container from a second, independent constant (a cells-per-outer-hit ratio) is a
    // redundant failure mode: it can be short while maxDoublets is not, and the shortfall then
    // silently truncates the hit->cell association rather than the cell list. One entry per cell,
    // one bound.
    // The handle and per-outer-hit offsets (key space) are always allocated.
    device_hitToCell_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToCellOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, outerHits + 1);
    if (!countFirst) {
      uint32_t nHitsToCells = std::max(cellBound, 1u);
      GenericContainerStorage *hitToCellStorage = arenaHitToCellStorage_;
      if (hitToCellStorage == nullptr) {
        device_hitToCellStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToCells);
        hitToCellStorage = device_hitToCellStorage_->data();
      }
      device_hitToCellView_ = {
          device_hitToCell_->data(), device_hitToCellOffsets_->data(), hitToCellStorage, outerHits + 1, nHitsToCells};

      HitToCell::template launchZero<Acc1D>(device_hitToCellView_, queue);
    }

    // Hits Phi Histograms: one histogram per layer
    device_hitPhiHist_ = cms::alpakatools::make_device_buffer<PhiBinner>(queue);
    device_phiBinnerStorage_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nHits);
    device_hitPhiView_ = {
        device_hitPhiHist_->data(), nullptr, device_phiBinnerStorage_->data(), cms::alpakatools::kDynamicSize, nHits};
    // This will hold where each layer starts in the hit soa
    device_layerStarts_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nLayers + 1);

    // Scratch quality mirror used by the (order-independent) duplicate-removal kernels
    device_qualityScratch_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, maxTuples);

    // Cell array. Allocated here, ahead of allocateAfterDoublets, because in the up-front allocation
    // mode it doubles as the arena that also carries the cell->track offsets (see the header).
    // With countDoubletsFirst the cells buffer is instead sized to the exact nCells in buildDoublets.
    if (!countFirst) {
      const uint32_t arenaExtent =
          (cellLayout == CellLayout::kCellsWithTrackOffsets) ? cellArenaExtent(cellBound, maxDoublets) : 0u;
      device_simpleCells_ =
          cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, arenaExtent ? arenaExtent : cellBound);
      if (arenaExtent) {
        const std::size_t cellsPadded = ((std::size_t(cellBound) + 63u) / 64u) * 64u;
        arenaCellToTracksOffsets_ =
            reinterpret_cast<GenericContainerOffsets *>(device_simpleCells_->data() + cellsPadded);
      }
    }

    // Cell -> Neighbor Cells and Cell -> Tracks (+ the triplet/track-cell SoA) are sized
    // from the number of doublets
    // With delayAllocations_ they are allocated in allocateAfterDoublets from the
    // actual nCells (after buildDoublets)
    // Otherwise they are allocated here at the maxDoublets safety cap by the same function
    // with nCells = maxDoublets
    if (!delay) {
      allocateAfterDoublets(maxDoublets, queue);
    }

    // Track -> Hits
    // - This is a OneToManyAssocSequential since each bin is filled
    //   in one go: all the hits forming a track are pushed together.
    device_hitContainer_ = cms::alpakatools::make_device_buffer<SequentialContainer>(queue);
    device_hitContainerStorage_ =
        cms::alpakatools::make_device_buffer<SequentialContainerStorage[]>(queue, nHitsToTracks);
    device_hitContainerOffsets_ =
        cms::alpakatools::make_device_buffer<SequentialContainerOffsets[]>(queue, maxTuples + 1);
    device_hitContainerView_ = {device_hitContainer_->data(),
                                device_hitContainerOffsets_->data(),
                                device_hitContainerStorage_->data(),
                                maxTuples + 1u,
                                nHitsToTracks};

    HitContainer::template launchZero<Acc1D>(device_hitContainerView_, queue);
    device_tupleClampBound_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2u);

    // No.Hits -> Track (track multiplicity)
    device_tupleMultiplicity_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_tupleMultiplicityStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, maxTuples);
    device_tupleMultiplicityOffsets_ =
        cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, TrackerTraits::maxHitsOnTrack + 2);
    device_tupleMultiplicityView_ = {
        device_tupleMultiplicity_->data(),
        device_tupleMultiplicityOffsets_->data(),
        device_tupleMultiplicityStorage_->data(),
        // this has to be +2 instead of +1 because you want all values from 0 to maxHitsOnTrack to be valid keys
        // (N+1 values) + the extra +1 for the Container definition
        TrackerTraits::maxHitsOnTrack + 2u,
        maxTuples};
    TupleMultiplicity::template launchZero<Acc1D>(device_tupleMultiplicityView_, queue);

    // Structures and Counters Storage. device_simpleCells_ was allocated above, before
    // allocateAfterDoublets, so that it can carry the cell->track offsets in its tail.
    device_extraStorage_ =
        cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 5u);
    device_hitTuple_apc_ = reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_extraStorage_->data());
    device_nCells_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 2));
    device_nTriplets_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 3));
    device_nCellTracks_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 4));

    // deviceTriplets_ and deviceTracksCells_ are sized from nCells in allocateAfterDoublets

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters for diagnostic funnel
    device_pipelineCounters_ =
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, caHitNtupletGenerator::kTotalCounters);
    alpaka::memset(queue, *device_pipelineCounters_, 0);
#endif

    //TODO: if doStats?
    alpaka::memset(queue, *counters_, 0);

    alpaka::memset(queue, *device_nCells_, 0);
    alpaka::memset(queue, *device_nTriplets_, 0);
    alpaka::memset(queue, *device_nCellTracks_, 0);

    maxNumberOfDoublets_ = cellBound;
    launchCells_ = cellBound;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Allocations for CAHitNtupletGeneratorKernels: done!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  uint32_t CAHitNtupletGeneratorKernels<TrackerTraits>::readbackNCells(Queue &queue) {
    using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
    auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
    alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
    alpaka::wait(queue);  // Necessary wait: the value is returned to the host caller, which uses it to
                          // size the next launches. A device->host readback the framework cannot order
                          // for us.
    // device_nCells_ aliases extraStorage word [2].
    return static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[2]);
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::enqueueCountsReadback(
      cms::alpakatools::host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]> &dst, Queue &queue) {
    // Async copy only -- the caller reads dst after its queue has been synchronized (the CA
    // producer's acquire->produce seam), so no host wait is needed here.
    alpaka::memcpy(queue, dst, *this->device_extraStorage_);
  }
  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::readbackAllCounts(Queue &queue,
                                                                      uint32_t &nTracks,
                                                                      uint32_t &nHitsInTracks,
                                                                      uint32_t &nCells,
                                                                      uint32_t &nTriplets,
                                                                      uint32_t &nCellTracks) {
    using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
    auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
    alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
    alpaka::wait(queue);
    // Word [0] is the AtomicPairCounter: its low half is `first` (the tuple count) and its high
    // half is `second` (the hits-in-tracks total), the same halves Kernel_overflowSentinel reads.
    // Taken by position, not by size: the two are only ordered while every tuple has >= 2 hits.
    const uint64_t apc_raw = static_cast<uint64_t>(alpaka::getPtrNative(h_extra)[0]);
    nTracks = static_cast<uint32_t>(apc_raw & 0xFFFFFFFFull);
    nHitsInTracks = static_cast<uint32_t>(apc_raw >> 32);
    // word [2] = nCells, [3] = nTriplets, [4] = nCellTracks
    nCells = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[2]);
    nTriplets = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[3]);
    nCellTracks = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[4]);
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::allocateAfterDoublets(uint32_t nCells, Queue &queue) {
    auto const &algoParams = m_params.algoParams_;

    // Association-content sizes scale with the number of doublets actually found
    // Keep the per-key avg multipliers but base them on nCells instead of the
    // (much larger) maxDoublets safety cap
    // Quiet-event floor. avgCellsPerCell_/avgTracksPerCell_ are averages fitted at high occupancy,
    // and an average is not a capacity bound: there most doublets are combinatorial junk with no
    // valid neighbour, so <cells per cell> is ~0.1, while in a clean event (a no-PU muon gun, say)
    // essentially every doublet is a real track segment with a genuine neighbour and the ratio rises
    // towards 1 while nCells, and hence the derived capacity, falls. The formula therefore shrinks
    // exactly where the requirement grows. The surplus would be dropped silently by the capacity
    // guards in Kernel_connect and CACell::find_ntuplets, and which entries survive is decided by
    // who wins the atomic: deterministic on the serial backend, race order on the GPU, i.e. an
    // efficiency loss that differs between backends and between runs.
    // The floor is the clean-event bound -- at most kCleanEdgesPerCell edges per cell -- tracked
    // linearly while the event is small enough for that to be affordable (kQuietCellsCap) and
    // constant above it. The quiet and the high-occupancy regimes are orders of magnitude apart in
    // nCells, which is what makes an absolute floor both sufficient and inert at high occupancy.
    // Events whose demand exceeds the floor (no-PU high-pT dijets are the worst case) truncate
    // through the counting guards and are reported by the overflow sentinel.
    constexpr uint32_t kQuietCellsCap = 512u;
    constexpr uint32_t kCleanEdgesPerCell = 64u;
    const uint32_t quietFloor = std::min(nCells, kQuietCellsCap) * kCleanEdgesPerCell;
    const uint32_t nCellsToCells = std::max({uint32_t(nCells * algoParams.avgCellsPerCell_), quietFloor, 1u});
    const uint32_t nCellsToTracks = std::max({uint32_t(nCells * algoParams.avgTracksPerCell_), quietFloor, 1u});
    const uint32_t nBins = nCells + 1u;  // one offset bin per cell (+1 for the total)

    // Offsets vs storage: the two extents computed above have different natures.
    //   nBins (the offsets extent of both associators) is a key-space extent: one word per cell plus
    //   the total. Its only consumers are the nOnes() bounds (Kernel_connect's cell loop and
    //   CACell::find_ntuplets), and every index they test is a cell id, hence < nCells by construction.
    //   nCellsToCells / nCellsToTracks (the storage extents) are content capacities: the truncation
    //   guards compare against them (Kernel_connect's maxTriplets test and the cell->track push in
    //   CACell::find_ntuplets), so entries past them are dropped silently. Their per-cell ratios are
    //   calibrated against the basis in force here; changing the basis without re-deriving the ratios
    //   moves a content-reaching clamp.
    // This function is called either once at construction with the maxDoublets safety cap -- no
    // doublet count exists yet at that point -- or after the doublet build with the exact count.

    // Cell -> Neighbor Cells. One bin per cell; bit 31 of each stored neighbor index
    // encodes layer-skipping (valid since nCells is well below 2^31).
    device_cellToNeighbors_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToNeighborsStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToCells);
    GenericContainerOffsets *cellToNeighborsOffsets = arenaCellToNeighborsOffsets_;
    if (cellToNeighborsOffsets == nullptr) {
      device_cellToNeighborsOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nBins);
      cellToNeighborsOffsets = device_cellToNeighborsOffsets_->data();
    }
    device_cellToNeighborsView_ = {device_cellToNeighbors_->data(),
                                   cellToNeighborsOffsets,
                                   device_cellToNeighborsStorage_->data(),
                                   nBins,
                                   nCellsToCells};
    CellToCell::template launchZero<Acc1D>(device_cellToNeighborsView_, queue);

    // Cell -> Tracks.
    device_cellToTracks_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToTracksStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToTracks);
    // The offsets live in the tail of the cell arena when the constructor set it up (up-front
    // allocation mode); otherwise they get their own allocation. Same extent, same contents either way.
    GenericContainerOffsets *cellToTracksOffsets = arenaCellToTracksOffsets_;
    if (cellToTracksOffsets == nullptr) {
      device_cellToTracksOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nBins);
      cellToTracksOffsets = device_cellToTracksOffsets_->data();
    }
    device_cellToTracksView_ = {
        device_cellToTracks_->data(), cellToTracksOffsets, device_cellToTracksStorage_->data(), nBins, nCellsToTracks};
    CellToTrack::template launchZero<Acc1D>(device_cellToTracksView_, queue);

    // Triplet (cell->cell) and track-cell (cell->track) SoA edge lists.
    tripletsN_ = nCellsToCells;
    tracksCellsN_ = nCellsToTracks;
    deviceTriplets_ = CAPairSoACollection(queue, tripletsN_);
    deviceTracksCells_ = CAPairSoACollection(queue, tracksCellsN_);
#ifdef CA_TRIPLET_DUMP
    device_tripletDump_ = TripletDumpSoACollection(queue, tripletsN_);
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::allocateAfterNtuplets(uint32_t nHitsInTracks, Queue &queue) {
    // Hit -> Track storage holds exactly nHitsInTracks entries (sum over surviving
    // tracks of their hit multiplicity)
    // With delayAllocations_ it is sized here from the real count
    // Otherwise it was already allocated in the constructor and this is a no-op
    // (apart from the GPU_DEBUG report below)
    // nHitsInTracks is 0 in that case
    if (m_params.algoParams_.delayAllocations_) {
      const uint32_t nStorage = std::max(nHitsInTracks, 1u);
      const uint32_t nHits = static_cast<uint32_t>(alpaka::getExtents(*device_hitToTupleOffsets_)[0u]) - 1u;
      device_hitToTupleStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nStorage);
      device_hitToTupleView_ = {device_hitToTuple_->data(),
                                device_hitToTupleOffsets_->data(),
                                device_hitToTupleStorage_->data(),
                                nHits + 1u,
                                nStorage};
      HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);
    }

#ifdef GPU_DEBUG
    // Per-buffer device allocation report. Runs here, after the last (deferred)
    // allocation, so every buffer's real extent is available
    {
      const auto coutFlags = std::cout.flags();
      const auto coutPrec = std::cout.precision();

      std::size_t total = 0;
      // Array buffers report their real element count; scalar "handle" buffers (the
      // OneToMany container objects) report 1 element when allocated.
      auto extentOf = [](auto const &optBuf) -> std::size_t {
        return optBuf ? static_cast<std::size_t>(alpaka::getExtents(*optBuf)[0u]) : 0u;
      };
      auto oneIf = [](auto const &optBuf) -> std::size_t { return optBuf ? 1u : 0u; };
      auto report = [&total](const char *name, std::size_t n_elem, std::size_t elem_size) {
        const std::size_t bytes = n_elem * elem_size;
        total += bytes;
        std::cout << "  " << std::left << std::setw(46) << name << " : " << std::right << std::setw(10) << n_elem
                  << " x " << std::setw(4) << elem_size << " B = " << std::setw(12) << bytes << " B  (" << std::fixed
                  << std::setprecision(3) << std::setw(9) << (bytes / 1024.0) << " KiB)" << std::endl;
      };
      auto reportSoA = [&total](const char *name, std::size_t n_elem, std::size_t bytes) {
        total += bytes;
        std::cout << "  " << std::left << std::setw(46) << name << " : " << std::right << std::setw(10) << n_elem
                  << " elements,  SoA layout = " << std::setw(12) << bytes << " B  (" << std::fixed
                  << std::setprecision(3) << std::setw(9) << (bytes / 1024.0) << " KiB)" << std::endl;
      };

      std::cout << "================== Device allocation report ==================" << std::endl;
      report("counters_", oneIf(counters_), sizeof(Counters));
      report("device_hitToTuple_", oneIf(device_hitToTuple_), sizeof(GenericContainer));
      report("device_hitToTupleStorage_", extentOf(device_hitToTupleStorage_), sizeof(GenericContainerStorage));
      report("device_hitToTupleOffsets_", extentOf(device_hitToTupleOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitToCell_", oneIf(device_hitToCell_), sizeof(GenericContainer));
      report("device_hitToCellStorage_", extentOf(device_hitToCellStorage_), sizeof(GenericContainerStorage));
      report("device_hitToCellOffsets_", extentOf(device_hitToCellOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitPhiHist_", oneIf(device_hitPhiHist_), sizeof(PhiBinner));
      report("device_phiBinnerStorage_", extentOf(device_phiBinnerStorage_), sizeof(PhiBinnerStorageType));
      report("device_layerStarts_", extentOf(device_layerStarts_), sizeof(hindex_type));
      report("device_qualityScratch_", extentOf(device_qualityScratch_), sizeof(int32_t));
      report("device_cellToNeighbors_", oneIf(device_cellToNeighbors_), sizeof(GenericContainer));
      report(
          "device_cellToNeighborsStorage_", extentOf(device_cellToNeighborsStorage_), sizeof(GenericContainerStorage));
      report(
          "device_cellToNeighborsOffsets_", extentOf(device_cellToNeighborsOffsets_), sizeof(GenericContainerOffsets));
      report("device_cellToTracks_", oneIf(device_cellToTracks_), sizeof(GenericContainer));
      report("device_cellToTracksStorage_", extentOf(device_cellToTracksStorage_), sizeof(GenericContainerStorage));
      report("device_cellToTracksOffsets_", extentOf(device_cellToTracksOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitContainer_", oneIf(device_hitContainer_), sizeof(SequentialContainer));
      report("device_hitContainerStorage_", extentOf(device_hitContainerStorage_), sizeof(SequentialContainerStorage));
      report("device_hitContainerOffsets_", extentOf(device_hitContainerOffsets_), sizeof(SequentialContainerOffsets));
      report("device_tupleMultiplicity_", oneIf(device_tupleMultiplicity_), sizeof(GenericContainer));
      report("device_tupleMultiplicityStorage_",
             extentOf(device_tupleMultiplicityStorage_),
             sizeof(GenericContainerStorage));
      report("device_tupleMultiplicityOffsets_",
             extentOf(device_tupleMultiplicityOffsets_),
             sizeof(GenericContainerOffsets));
      report("device_simpleCells_", extentOf(device_simpleCells_), sizeof(SimpleCell));
      report("device_extraStorage_",
             extentOf(device_extraStorage_),
             sizeof(cms::alpakatools::AtomicPairCounter::DoubleWord));
#ifdef CA_PIPELINE_COUNTERS
      report("device_pipelineCounters_", extentOf(device_pipelineCounters_), sizeof(uint32_t));
#endif

      const std::size_t tripletsBytes = CAPairSoACollection::Layout::computeDataSize(tripletsN_);
      const std::size_t tracksCellsBytes = CAPairSoACollection::Layout::computeDataSize(tracksCellsN_);
      reportSoA("deviceTriplets_ (SoA)", tripletsN_, tripletsBytes);
      reportSoA("deviceTracksCells_ (SoA)", tracksCellsN_, tracksCellsBytes);

      std::cout << "  --------------------------------------------------------------" << std::endl;
      std::cout << "  " << std::left << std::setw(46) << "THIS EVENT'S TOTAL"
                << " : " << std::right << std::setw(12) << total << " B  (" << std::fixed << std::setprecision(3)
                << std::setw(9) << (total / (1024.0 * 1024.0)) << " MiB)" << std::endl;

      uint64_t snapAllocN = 0;
      uint64_t snapAllocSum = 0;
      uint64_t snapAllocMax = 0;
      {
        auto &agg = sizingAccumulator();
        std::lock_guard<std::mutex> lock(agg.mtx);
        ++agg.allocN;
        agg.allocBytesSum += static_cast<uint64_t>(total);
        if (static_cast<uint64_t>(total) > agg.allocBytesMax)
          agg.allocBytesMax = static_cast<uint64_t>(total);
        snapAllocN = agg.allocN;
        snapAllocSum = agg.allocBytesSum;
        snapAllocMax = agg.allocBytesMax;
      }
      const double meanBytes = double(snapAllocSum) / double(snapAllocN);
      std::cout << "  " << std::left << std::setw(46) << "AGGREGATE (mean over events)"
                << " : " << std::right << std::setw(12) << static_cast<uint64_t>(meanBytes) << " B  (" << std::fixed
                << std::setprecision(3) << std::setw(9) << (meanBytes / (1024.0 * 1024.0)) << " MiB)  over "
                << snapAllocN << " event" << (snapAllocN == 1u ? "" : "s") << std::endl;
      std::cout << "  " << std::left << std::setw(46) << "AGGREGATE (max over events)"
                << " : " << std::right << std::setw(12) << snapAllocMax << " B  (" << std::fixed << std::setprecision(3)
                << std::setw(9) << (snapAllocMax / (1024.0 * 1024.0)) << " MiB)" << std::endl;
      std::cout << "==============================================================" << std::endl;

      std::cout.flags(coutFlags);
      std::cout.precision(coutPrec);
    }
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::prepareHits(const HitsMultiView &hh,
                                                                const ModulesMultiView &mm,
                                                                const reco::CALayersSoAConstView &ll,
                                                                Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;

    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, ll.metadata().size() - 1);
    alpaka::exec<Acc1D>(
        queue, workDiv1D, SetHitsLayerStart<ModulesMultiView>{}, mm, ll, this->device_layerStarts_->data());

    auto accessor_iphi = [] ALPAKA_FN_ACC(auto const &v) { return v.iphi(); };

    cms::alpakatools::fillManyFromVector<Acc1D>(device_hitPhiHist_->data(),
                                                device_hitPhiView_,
                                                TrackerTraits::numberOfLayers,  // could be ll.metadata().size() - 1
                                                hh,
                                                accessor_iphi,
                                                this->device_layerStarts_->data(),
                                                static_cast<uint32_t>(hh.size()),
                                                static_cast<uint32_t>(256),
                                                queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "CAHitNtupletGeneratorKernels -> Hits prepared (layer starts and histo) -> DONE!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::launchKernels(const HitsMultiView &hh,
                                                                  uint32_t offsetBPIX2,
                                                                  uint16_t nLayers,
                                                                  TkSoABlocksView &view,
                                                                  const reco::CALayersSoAConstView &ll,
                                                                  const reco::CAGraphSoAConstView &cc,
                                                                  const reco::CATripletCutsSoAConstView &tripletCuts,
                                                                  const reco::CANtupletCutsSoAConstView &ntupletCuts,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    auto tracks_view = view.tracks();
    auto tracks_hits_view = view.trackHits();

    uint32_t nhits = static_cast<uint32_t>(hh.size());
    auto const launchCells = this->launchCells_;
    auto const maxTuples = tracks_view.metadata().size();
#ifdef NTUPLE_DEBUG
    std::cout << "start tuple building. N hits " << nhits << std::endl;
    if (nhits < 2)
      std::cout << "too few hits " << nhits << std::endl;
#endif

    //
    // applying combinatoric cleaning such as fishbone at this stage is too expensive
    //

    const auto nthTot = 64;
    const auto stride = 4;
    auto blockSize = nthTot / stride;
    // The cell/tuple kernels here and below launch over the FULL container capacity. The
    // loops inside are bounded by the true counts, so coverage never relies on the
    // grid-stride wrap-around -- and with the extent at least the capacity, each index is
    // visited by exactly one thread, which keeps the serial backend's visit order (and so
    // its bit-exact output) independent of the capacity value.
    //
    // Grow blockSize (in multiples of 16, keeping blockSize*stride <= 1024) until the grid
    // fits CUDA's 65535-block limit, whatever maxDoublets is.
    auto numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    while (numberOfBlocks >= 65536 && blockSize * stride < 1024) {
      blockSize += 16;
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    }
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks{numberOfBlocks, 1u};
    const Vec2D thrs{blockSize, stride};
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

    alpaka::exec<Acc2D>(queue,
                        kernelConnectWorkDiv,
                        Kernel_connect<TrackerTraits>{},
                        this->device_hitTuple_apc_,  // needed only to be reset, ready for next kernel
                        hh,
                        cc,
                        tripletCuts,
                        this->m_params.algoParams_.useTripletDNN_,
                        this->m_params.algoParams_.tripletDNNThreshold_,
#ifdef CA_TRIPLET_DUMP
                        this->device_tripletDump_->view(),
#endif
                        this->deviceTriplets_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_hitToCell_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->pipelineCountersPtr());

    CellToCell::template launchFinalize<Acc1D>(this->device_cellToNeighborsView_, queue);

#ifdef CA_TRIPLET_DUMP
    // Stamp the valid-row count into the dump SoA scalar (== *device_nTriplets_, the number of rows
    // Kernel_connect actually wrote). Device->device copy on the same queue; the host consumer reads
    // view().nValid() to know how many of the full-capacity rows are valid. Zero footprint when off.
    alpaka::memcpy(queue,
                   cms::alpakatools::make_device_view(queue, this->device_tripletDump_->view().nValid()),
                   cms::alpakatools::make_device_view(queue, *this->device_nTriplets_->data()));
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_connect -> Done!" << std::endl;
#endif

    auto threadsPerBlock = 1024;
    // Floor at one block: with useExactAllocations the launch cell count can be a handful, and
    // lrint(nCells * avgCellsPerCell) then rounds to 0, i.e. a zero-block launch that fills nothing.
    auto blocks = std::max<cms::alpakatools::Idx>(
        1u,
        cms::alpakatools::divide_up_by(std::lrint(launchCells * m_params.algoParams_.avgCellsPerCell_),
                                       threadsPerBlock));
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillGenericPair<caStructures::CAPairSoAConstView, GenericContainer>{},
                        this->deviceTriplets_->view(),
                        this->device_nTriplets_->data(),
                        this->device_cellToNeighbors_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "cellToNeighbors -> Filled!" << std::endl;
#endif

    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.algoParams_.earlyFishbone_ and nhits > offsetBPIX2) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits - offsetBPIX2, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto fishboneWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
      alpaka::exec<Acc2D>(queue,
                          fishboneWorkDiv,
                          CAFishbone<TrackerTraits>{},
                          hh,
                          ll,
                          cc,
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_hitToCell_->data(),
                          this->device_cellToTracks_->data(),
                          nhits - offsetBPIX2,
                          false,
                          this->pipelineCountersPtr(),
                          this->m_params.algoParams_.onlySameLayersFishbone_);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Early fishbone -> Done!" << std::endl;
#endif
    }

#ifdef CA_PIPELINE_COUNTERS
    // Count cell status after the fishbone kill phase
    {
      auto cellBlocks = cms::alpakatools::divide_up_by(launchCells, 256u);
      auto cellWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(cellBlocks, 256u);
      alpaka::exec<Acc1D>(queue,
                          cellWorkDiv,
                          Kernel_pipelineCellStatus<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->pipelineCountersPtr());
    }
#endif

    blockSize = 64;
    numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_find_ntuplets<TrackerTraits>{},
                        hh,
                        cc,
                        ntupletCuts,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->device_cellToTracks_->data(),
                        this->deviceTracksCells_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCellTracks_->data(),
                        this->device_nTriplets_->data(),
                        this->device_nCells_->data(),
                        this->device_hitTuple_apc_,
                        this->m_params.algoParams_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_find_ntuplets -> Done!" << std::endl;
#endif

#ifdef CA_PIPELINE_COUNTERS
    // Copy *nCellTracks into the pipeline counter array
    {
      auto workDiv1x1 = cms::alpakatools::make_workdiv<Acc1D>(1u, 1u);
      alpaka::exec<Acc1D>(queue,
                          workDiv1x1,
                          Kernel_pipelineCopyCellTrackCount{},
                          this->device_nCellTracks_->data(),
                          this->pipelineCountersPtr());
    }
#endif

    CellToTracks::template launchFinalize<Acc1D>(this->device_cellToTracksView_, queue);

    // This pass fills the cell->TRACK edge list, so its grid is sized from avgTracksPerCell_.
    // The loop inside is a grid-stride uniform_elements over the true edge count, so the block
    // count is a throughput choice only and can never truncate work.
    blocks = std::max<cms::alpakatools::Idx>(
        1u,
        cms::alpakatools::divide_up_by(std::lrint(launchCells * m_params.algoParams_.avgTracksPerCell_),
                                       threadsPerBlock));
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillGenericPair<caStructures::CAPairSoAConstView, GenericContainer>{},
                        this->deviceTracksCells_->view(),
                        this->device_nCellTracks_->data(),
                        this->device_cellToTracks_->data());

    if (this->m_params.algoParams_.doStats_)
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_mark_used<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_cellToTracks_->data(),
                          this->device_nCells_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples + 1, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        typename HitContainer::finalizeBulk{},
                        this->device_hitTuple_apc_,
                        this->device_hitContainer_->data());

    // Repair the CSR after a hit-container content overflow (see Kernel_findTupleContentOverflow):
    // overflowed tuples become empty instead of describing unwritten content and nTracks is cut
    // in front of them. The memset value 0xFFFFFFFF means "no clamp" and is what the no-overflow
    // case leaves in place.
    alpaka::memset(queue, *this->device_tupleClampBound_, 0xFF);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_findTupleContentOverflow{},
                        this->device_hitContainer_->data(),
                        this->device_tupleClampBound_->data());
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_clampTupleContentOverflow{},
                        this->device_hitContainer_->data(),
                        this->device_tupleClampBound_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline counter: classify n-tuplets by OT hit content.
    // Must run AFTER finalizeBulk so that foundNtuplets offsets are valid
    // for size()/begin()/end() iteration.
    {
      auto ntupBlocks = cms::alpakatools::divide_up_by(maxTuples, 128u);
      auto ntupWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(ntupBlocks, 128u);
      alpaka::exec<Acc1D>(queue,
                          ntupWorkDiv,
                          Kernel_pipelineNtupletCount<TrackerTraits>{},
                          hh,
                          this->device_hitContainer_->data(),
                          this->device_hitTuple_apc_,
                          maxTuples,
                          this->pipelineCountersPtr());
    }
#endif

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillHitDetIndices<TrackerTraits>{},
                        tracks_view,
                        tracks_hits_view,
                        this->device_hitContainer_->data(),
                        hh,
                        this->device_hitTuple_apc_,
                        this->device_tupleClampBound_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillHitDetIndices   -> done!" << std::endl;
#endif

    // remove duplicates (tracks that share a doublet).
    if (this->m_params.algoParams_.doEarlyDuplicateRemover_) {
      if constexpr (std::is_same_v<pixelTopology::Phase1, TrackerTraits>) {
        // for Phase-1, the workdivision is simpler and therefore needs a separate call here
        blockSize = 64;
        numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_earlyDuplicateRemoverPhase1{},
                            this->device_simpleCells_->data(),
                            this->device_nCells_->data(),
                            this->device_cellToTracks_->data(),
                            tracks_view,
                            this->m_params.algoParams_.dupPassThrough_);
      } else {
        // Warp-cooperative work division: one cell per Y-thread, one warp per cell.
        const uint32_t stride = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
        const uint32_t cellsPerBlock = 8;  // 8 cells * 32 threads = 256 threads/block
        // gridDim.y is capped at 65535 by CUDA. If maxDoublets needs more, uniform_elements_y
        // strides each Y-thread across the remaining cells.
        auto earlyDupRemoverNumBlocks = cms::alpakatools::divide_up_by(launchCells, cellsPerBlock);
        // Cap at Y-dim limit
        earlyDupRemoverNumBlocks = std::min<uint32_t>(earlyDupRemoverNumBlocks, 65535u);
        const Vec2D earlyDupRemoverBlocks{earlyDupRemoverNumBlocks, 1u};
        const Vec2D earlyDupRemoverThreads{cellsPerBlock, stride};
        const auto earlyDupRemoverWorkDiv =
            cms::alpakatools::make_workdiv<Acc2D>(earlyDupRemoverBlocks, earlyDupRemoverThreads);

        alpaka::exec<Acc2D>(queue,
                            earlyDupRemoverWorkDiv,
                            Kernel_earlyDuplicateRemover<TrackerTraits>{},
                            this->device_simpleCells_->data(),
                            this->device_nCells_->data(),
                            this->device_cellToTracks_->data(),
                            tracks_view,
                            this->m_params.algoParams_.dupPassThrough_);
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_earlyDuplicateRemover   -> done!" << std::endl;
#endif
    }

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_countMultiplicity<TrackerTraits>{},
                        hh,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_tupleMultiplicity_->data());
    GenericContainer::template launchFinalize<Acc1D>(this->device_tupleMultiplicityView_, queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_countMultiplicity   -> done!" << std::endl;
#endif

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillMultiplicity<TrackerTraits>{},
                        hh,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_tupleMultiplicity_->data());
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillMultiplicity -> done!" << std::endl;
#endif
    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.algoParams_.lateFishbone_ and nhits > offsetBPIX2) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits - offsetBPIX2, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          CAFishbone<TrackerTraits>{},
                          hh,
                          ll,
                          cc,
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_hitToCell_->data(),
                          this->device_cellToTracks_->data(),
                          nhits - offsetBPIX2,
                          true,
                          this->pipelineCountersPtr(),
                          this->m_params.algoParams_.onlySameLayersFishbone_);
    }

#ifdef GPU_DEBUG
    std::cout << "lateFishbone -> done!" << std::endl;
    alpaka::wait(queue);
#endif

    // Every build-only buffer has now had its last reader enqueued. Hand them back here instead of at
    // the end of produce(): tens of MiB return to the caching allocator half an event early, which
    // lets another stream reuse the bin instead of growing the pool.
    this->releaseBuildScratch();
  }

  // Early release of the build-only scratch (called at the end of launchKernels). No new
  // synchronisation: cms::alpakatools::CachingAllocator::free is stream-ordered, it records an event
  // on the queue the block was allocated against and only re-issues the block once that event has
  // completed, so the kernels already enqueued keep valid pointers for as long as they need them.
  // For each buffer below the last reader is a kernel enqueued in launchKernels or earlier:
  //   device_hitToCell_ (+Storage/Offsets)        last read by the late-fishbone CAFishbone launch
  //                                               (classifyTuples' shared-hit cleaner uses
  //                                               device_hitToTuple_, a different container);
  //   device_cellToNeighbors_ (+Storage/Offsets)  last read by Kernel_find_ntuplets. The cell->TRACKS
  //                                               container is NOT released: the duplicate remover
  //                                               reads it in classifyTuples;
  //   device_hitPhiHist_ / device_phiBinnerStorage_  read only by buildDoublets' doublet kernels;
  //   deviceTriplets_ / deviceTracksCells_        afterwards only their capacity is wanted, which is the
  //                                               host-side tripletsN_/tracksCellsN_ (read by the
  //                                               overflow sentinel). Kernel_checkOverflows reads them
  //                                               on device under doStats_, so then they stay alive.
  // The allocation modes only move where the extents come from, before launchKernels. The views that
  // alias these buffers are not read again either; the object lives for one event.
  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::releaseBuildScratch() {
#if defined(GPU_DEBUG) || defined(CA_STATS)
    // Diagnostic builds keep everything to the end of the event: allocateAfterNtuplets' per-buffer
    // allocation report and the CA_STATS outerHits recovery read these buffers' real extents after
    // launchKernels has returned.
    return;
#else
    device_hitToCellStorage_.reset();
    device_hitToCellOffsets_.reset();
    device_hitToCell_.reset();

    device_cellToNeighborsStorage_.reset();
    device_cellToNeighborsOffsets_.reset();
    device_cellToNeighbors_.reset();

    device_phiBinnerStorage_.reset();
    device_hitPhiHist_.reset();

    if (!m_params.algoParams_.doStats_) {
      deviceTriplets_.reset();
      deviceTracksCells_.reset();
    }
#endif
  }

  template <typename TrackerTraits>
  uint32_t CAHitNtupletGeneratorKernels<TrackerTraits>::buildDoublets(
      const HitsMultiView &hh,
      const ::reco::CAGraphSoAConstView &cc,
      const ::reco::CALayersSoAConstView &ll,
      const ::reco::CADoubletCutsSoAConstView &doubletCuts,
      uint32_t offsetBPIX2,
      Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    auto nhits = hh.size();
    const auto maxDoublets = this->maxNumberOfDoublets_;
#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    if (0 == nhits)
      return 0u;  // protect against empty events

    const int stride = 4;
    int threadsPerBlock = TrackerTraits::getDoubletsFromHistoMaxBlockSize / stride;
    int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
    const Vec2D blks{blocks, 1u};
    const Vec2D thrs{threadsPerBlock, stride};
    const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

#ifdef GPU_DEBUG
    std::cout << "nActualPairs = " << cc.metadata().size() << std::endl;
    std::cout << blocks << " - " << threadsPerBlock << " - " << stride << std::endl;
#endif
    // The exact doublet count, when countDoubletsFirst_ paid a readback for it. Returned to the
    // caller so allocateAfterDoublets can be sized from it without a second D2H of the same word.
    uint32_t nCellsCounted = 0u;

    if (this->m_params.algoParams_.countDoubletsFirst_) {
      // countDoubletsFirst_ -> do a dry-run of GetDoubletsFromHisto to learn the exact nCells
      // without writing cells or the hit->cell association, then size device_simpleCells_ and device_hitToCellStorage_
      // to that count before the real fill pass
      alpaka::memset(queue, *this->device_nCells_, 0);
      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          GetDoubletsFromHisto<TrackerTraits, /*CountOnly=*/true>{},
                          maxDoublets,
                          static_cast<SimpleCell *>(nullptr),
                          this->device_nCells_->data(),
                          hh,
                          cc,
                          ll,
                          doubletCuts,
                          this->device_layerStarts_->data(),
                          this->device_hitPhiHist_->data(),
                          this->device_hitToCell_->data(),
                          this->pipelineCountersPtr());

      const uint32_t nCellsFound = this->readbackNCells(queue);
      nCellsCounted = nCellsFound;
      const uint32_t nCellsStorage = std::max(nCellsFound, 1u);
      this->launchCells_ = nCellsStorage;
      const uint32_t outerHits = static_cast<uint32_t>(alpaka::getExtents(*this->device_hitToCellOffsets_)[0u]) - 1u;

      this->device_simpleCells_ = cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, nCellsStorage);
      this->device_hitToCellStorage_ =
          cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsStorage);
      this->device_hitToCellView_ = {this->device_hitToCell_->data(),
                                     this->device_hitToCellOffsets_->data(),
                                     this->device_hitToCellStorage_->data(),
                                     outerHits + 1u,
                                     nCellsStorage};
      HitToCell::template launchZero<Acc1D>(this->device_hitToCellView_, queue);

      alpaka::memset(queue, *this->device_nCells_, 0);  // reset for the fill pass
    }

    // Fill the cells and the hit->cell association
    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        GetDoubletsFromHisto<TrackerTraits>{},
                        maxDoublets,
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        hh,
                        cc,
                        ll,
                        doubletCuts,
                        this->device_layerStarts_->data(),
                        this->device_hitPhiHist_->data(),
                        this->device_hitToCell_->data(),
                        this->pipelineCountersPtr());

    HitToCell::template launchFinalize<Acc1D>(this->device_hitToCellView_, queue);

    threadsPerBlock = 512;
    blocks = cms::alpakatools::divide_up_by(this->launchCells_, threadsPerBlock);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "GetDoubletsFromHisto   -> done!" << std::endl;
#endif

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        FillDoubletsHisto<TrackerTraits>{},
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        offsetBPIX2,
                        this->device_hitToCell_->data(),
                        this->counters_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "FillDoubletsHisto   -> done!" << std::endl;
#endif

    return nCellsCounted;
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples(const HitsMultiView &hh,
                                                                   TkSoAView &tracks_view,
                                                                   Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples" << std::endl;
#endif

    const uint32_t nhits = static_cast<uint32_t>(hh.size());

    auto blockSize = 64;
    const uint32_t launchCells = this->launchCells_;
    const uint32_t maxTuples = tracks_view.metadata().size();

    // Order/backend-independent duplicate removal via the int32 quality scratch (see the kernel headers)
    // snapshotQuality() freezes quality() into the scratch; applyQuality() copies it back. The fast
    // remover needs both; the track-parallel cleaners are single-writer and need only the snapshot
    auto qScratch = this->device_qualityScratch_->data();
    auto qScratchWorkDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(maxTuples, blockSize), blockSize);
    auto snapshotQuality = [&]() {
      alpaka::exec<Acc1D>(
          queue, qScratchWorkDiv, Kernel_snapshotQuality{}, tracks_view, this->device_hitContainer_->data(), qScratch);
    };
    auto applyQuality = [&]() {
      alpaka::exec<Acc1D>(
          queue, qScratchWorkDiv, Kernel_applyQuality{}, tracks_view, this->device_hitContainer_->data(), qScratch);
    };

    // classify tracks based on kinematics
    auto numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_classifyTracks<TrackerTraits>{},
                        tracks_view,
                        this->device_hitContainer_->data(),
                        hh,
                        this->m_params.qualityCuts_,
                        this->m_params.algoParams_.useTrackDNN_,
                        this->m_params.algoParams_.trackDNNThreshold_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_classifyTracks -> done!" << std::endl;
#endif

    if (this->m_params.algoParams_.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fishboneCleaner<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToTracks_->data(),
                          tracks_view);
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fishboneCleaner   -> done!" << std::endl;
#endif
    if (this->m_params.algoParams_.doFastDuplicateRemover_) {
      // mark duplicates (tracks that share a doublet)
      // Two-tier work division (see the comment above Kernel_fastDuplicateRemover): one cell per
      // thread, with the whole warp ganging up on the rare cells whose track list is longer than
      // kDupCoopMinTracks. The block size MUST stay a multiple of the warp size: the kernel issues
      // full-mask warp collectives, and its grid-stride loop is lane-aligned only if the grid
      // stride is.
      blockSize = 64;
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      snapshotQuality();
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fastDuplicateRemover<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToTracks_->data(),
                          tracks_view,
                          qScratch,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->m_params.algoParams_.fastDupNSigma2_);
      applyQuality();
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_fastDuplicateRemover   -> done!" << std::endl;
#endif
    }
    if (this->m_params.algoParams_.doSharedHitCut_ || this->m_params.algoParams_.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data(),
                          nhits);  // tagged OT extras bin at nhits + otIdx

      GenericContainer::template launchFinalize<Acc1D>(this->device_hitToTupleView_, queue);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data(),
                          nhits,  // tagged OT extras bin at nhits + otIdx
                          this->counters_->data());

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_countHitInTracks   -> done!" << std::endl;
#endif
    }

    if (this->m_params.algoParams_.doSharedHitCut_) {
      // mark duplicates (tracks that share at least one hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      // The shared-hit removers are track-parallel and single-writer: snapshotQuality() freezes the
      // quality into the scratch, the kernel reads the scratch and writes only each track's own
      // quality directly (no atomics, no copy-back)
      snapshotQuality();
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_rejectDuplicate<TrackerTraits>{},
                          tracks_view,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitContainer_->data(),
                          qScratch,
                          this->device_hitToTuple_->data(),
                          this->m_params.algoParams_.fastDupNSigma2_);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_rejectDuplicate   -> done!" << std::endl;
#endif
      // Only run if the cut is doing something
      if (this->m_params.algoParams_.minHitsForSharingCut_ > 1) {
        snapshotQuality();
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_sharedHitCleaner<TrackerTraits>{},
                            hh,
                            this->device_layerStarts_->data(),
                            tracks_view,
                            this->m_params.algoParams_.minHitsForSharingCut_,
                            this->m_params.algoParams_.dupPassThrough_,
                            this->device_hitContainer_->data(),
                            qScratch,
                            this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "Kernel_sharedHitCleaner   -> done!" << std::endl;
#endif
      }

      if (this->m_params.algoParams_.doTripletCleaner_ && (this->m_params.algoParams_.minHitsPerNtuplet_ <= 3)) {
        if (this->m_params.algoParams_.useSimpleTripletCleaner_) {
          numberOfBlocks =
              cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
          workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
          snapshotQuality();
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              Kernel_simpleTripletCleaner<TrackerTraits>{},
                              tracks_view,
                              this->m_params.algoParams_.dupPassThrough_,
                              this->device_hitContainer_->data(),
                              qScratch,
                              this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_simpleTripletCleaner   -> done!" << std::endl;
#endif
        } else {
          numberOfBlocks =
              cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
          workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
          snapshotQuality();
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              Kernel_tripletCleaner<TrackerTraits>{},
                              tracks_view,
                              this->m_params.algoParams_.dupPassThrough_,
                              this->device_hitContainer_->data(),
                              qScratch,
                              this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_tripletCleaner   -> done!" << std::endl;
#endif
        }
      }
    }

    if (this->m_params.algoParams_.doStats_) {
      numberOfBlocks = cms::alpakatools::divide_up_by(std::max(nhits, launchCells), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_checkOverflows<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_tupleMultiplicity_->data(),
                          this->device_hitToTuple_->data(),
                          this->device_hitTuple_apc_,
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_nTriplets_->data(),
                          this->device_nCellTracks_->data(),
                          this->deviceTriplets_->view(),
                          this->deviceTracksCells_->view(),
                          nhits,
                          this->maxNumberOfDoublets_,
                          this->m_params.algoParams_,
                          this->counters_->data());
    }

    // Always-on overflow sentinel: doStats-independent surface for the capacity guards (see the
    // kernel comment). The accumulator is per-stream persistent, owned by CAHitNtupletGenerator;
    // null when no owner armed it (non-CA users of this class).
    if (this->ovfAccum_ != nullptr) {
      auto sentinelDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(queue,
                          sentinelDiv,
                          Kernel_overflowSentinel{},
                          this->device_hitTuple_apc_,
                          this->device_nCells_->data(),
                          this->device_nTriplets_->data(),
                          this->device_nCellTracks_->data(),
                          uint32_t(tracks_view.metadata().size()),
                          this->maxNumberOfDoublets_,
                          // The two CA pair-SoA capacities. These are the HOST-SIDE element counts the
                          // collections were constructed from in allocateAfterDoublets, so reading
                          // them off the class is the same number the views' metadata().size()
                          // returned -- and it survives the build-only buffers being released at the
                          // end of launchKernels.
                          this->tripletsN_,
                          this->tracksCellsN_,
                          // hitContainer content slots (allocated in the ctor, extent host-known).
                          // This is the binding per-track-hit bound: the output trackHits SoA is
                          // sized from the same expression, so it is never the smaller of the two.
                          uint32_t(alpaka::getExtents(*this->device_hitContainerStorage_)[0u]),
                          // hitToTuple content slots. Under delayed allocation the storage is sized
                          // from the hits-in-tracks readback, i.e. to exactly the demand scalar this
                          // check compares against, so the comparison carries no information and
                          // 0xFFFFFFFF disables it (same value used when the storage is absent).
                          (this->m_params.algoParams_.delayAllocations_ || !this->device_hitToTupleStorage_)
                              ? 0xFFFFFFFFu
                              : uint32_t(alpaka::getExtents(*this->device_hitToTupleStorage_)[0u]),
                          this->ovfAccum_);
    }

#ifdef CA_STATS
    alpaka::wait(queue);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_printSizes<HitsMultiView>{},
                        hh,
                        tracks_view,
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_nCellTracks_->data());

    alpaka::wait(queue);

    // Report the per-event statistics and recommendations for sizing the avg parameters.
    // Each parameter sizes a storage buffer as cap = max_keys * avg.
    // The buffer is safe iff actual_fill <= cap, i.e.
    //     avg >= actual_fill / max_keys.
    //
    // After kernel execution the relevant fills are:
    //
    //   avgHitsPerTrack  -> fill = Sum_t nHits(t)  (= APC.n)
    //                       max_keys = maxTuples
    //   avgCellsPerHit   -> fill = nCells          (each cell has exactly 1 outer hit)
    //                       max_keys = outerHits
    //   avgCellsPerCell  -> fill = nTriplets       (one neighbor-link per triplet)
    //                       max_keys = maxDoublets
    //   avgTracksPerCell -> fill = nCellTrackPairs
    //                       max_keys = maxDoublets
    //
    // All four counters live in device_extraStorage_:
    //   [0,1] = AtomicPairCounter (m = nTracksFound, n = nHitsInTracks)
    //   [2]   = nCells
    //   [3] = nTriplets
    //   [4] = nCellTrackPairs
    std::cout << "========== CA Tracking Summary ==========" << std::endl;
    {
      using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
      auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
      alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
      alpaka::wait(queue);

      DW const *const extra = alpaka::getPtrNative(h_extra);

      // Decode the AtomicPairCounter that lives in extra[0]. The struct
      // packs (m, n) into a single 64-bit word. Extract apc_lo and apc_hi
      // as the two 32-bit halves and then use the smaller as the number of tracks
      // and the larger as the number of hits in tracks, since every track has at least one hit.
      const uint64_t apc_raw = static_cast<uint64_t>(extra[0]);
      const uint32_t apc_lo = static_cast<uint32_t>(apc_raw & 0xFFFFFFFFull);
      const uint32_t apc_hi = static_cast<uint32_t>(apc_raw >> 32);
      const uint32_t nTracksFnd = std::min(apc_lo, apc_hi);
      const uint32_t nHitsInTrk = std::max(apc_lo, apc_hi);
      const uint32_t nCellsF = static_cast<uint32_t>(extra[2]);
      const uint32_t nTripletsF = static_cast<uint32_t>(extra[3]);
      const uint32_t nCTPairsF = static_cast<uint32_t>(extra[4]);

      // outerHits is not a parameter of classifyTuples and is not stored
      // on the class. Recover it from device_hitToCellOffsets_
      const uint32_t outerHitsR = static_cast<uint32_t>(alpaka::getExtents(*this->device_hitToCellOffsets_)[0u]) - 1u;
      auto const &a = this->m_params.algoParams_;

      // Required (this event would not have overflowed if avg >= this). Each ratio MUST be
      // referenced to the SAME basis the buffer is actually allocated against. For the two
      // cell-keyed buffers that basis DEPENDS ON THE ALLOCATION MODE:
      //   - countDoubletsFirst / delayAllocations -> allocateAfterDoublets(nCells)  basis = nCells
      //   - neither (allocate up front)           -> allocateAfterDoublets(maxDoublets) basis = maxDoublets
      // Rather than re-derive which lane ran, scale the CURRENT avg by the buffer's actual
      // occupancy: capacity tripletsN_ = basis * avg_cur, so to fit nTriplets we need
      //   avg_req = avg_cur * nTriplets / tripletsN_   (mode-independent, correct in both lanes).
      const float req_HpT = (maxTuples > 0) ? float(nHitsInTrk) / float(maxTuples) : 0.f;
      const float req_CpH = (outerHitsR > 0) ? float(nCellsF) / float(outerHitsR) : 0.f;
      const float req_CpC = (tripletsN_ > 0u) ? a.avgCellsPerCell_ * float(nTripletsF) / float(tripletsN_) : 0.f;
      const float req_TpC = (tracksCellsN_ > 0u) ? a.avgTracksPerCell_ * float(nCTPairsF) / float(tracksCellsN_) : 0.f;

      // Physical mean per *actual* key (informative, NOT the sizing bound):
      const float phy_HpT = (nTracksFnd > 0) ? float(nHitsInTrk) / float(nTracksFnd) : 0.f;
      const float phy_CpH = req_CpH;  // each cell has exactly one outer hit
      const float phy_CpC = (nCellsF > 0) ? float(nTripletsF) / float(nCellsF) : 0.f;
      const float phy_TpC = (nCellsF > 0) ? float(nCTPairsF) / float(nCellsF) : 0.f;

      // Safety factor to provide some headroom
      constexpr float kSafety = 1.25f;

      // Pack into arrays so per-event and aggregate code can share a loop.
      const char *paramNames[4] = {"avgHitsPerTrack", "avgCellsPerHit", "avgCellsPerCell", "avgTracksPerCell"};
      const float currents[4] = {a.avgHitsPerTrack_, a.avgCellsPerHit_, a.avgCellsPerCell_, a.avgTracksPerCell_};
      const float req[4] = {req_HpT, req_CpH, req_CpC, req_TpC};
      const float phy[4] = {phy_HpT, phy_CpH, phy_CpC, phy_TpC};

      // Parameters that scale with nHits
      //   ratio[0] = nCells  / nHits  -> maxNumberOfDoublets
      //   ratio[1] = nTuples / nHits  -> maxNumberOfTuples
      const char *scaleNames[2] = {"maxNumberOfDoublets", "maxNumberOfTuples"};
      const uint32_t scaleCurrent[2] = {this->maxNumberOfDoublets_, maxTuples};
      const uint32_t scaleCount[2] = {nCellsF, nTracksFnd};
      const double ratio[2] = {(nhits > 0) ? double(nCellsF) / double(nhits) : 0.,
                               (nhits > 0) ? double(nTracksFnd) / double(nhits) : 0.};

      // Update the cross-event accumulator and take a snapshot under the
      // lock; print outside the lock to keep the critical section small.
      uint64_t snapN = 0;
      double snapMaxReq[4] = {0., 0., 0., 0.};
      double snapMeanReq[4] = {0., 0., 0., 0.};
      double snapMaxPhy[4] = {0., 0., 0., 0.};
      double snapMaxRatio[2] = {0., 0.};
      double snapMeanRatio[2] = {0., 0.};
      uint32_t snapMaxNHits = 0u;
      {
        auto &agg = sizingAccumulator();
        std::lock_guard<std::mutex> lock(agg.mtx);
        ++agg.n;
        for (int i = 0; i < 4; ++i) {
          if (req[i] > agg.maxReq[i])
            agg.maxReq[i] = req[i];
          if (phy[i] > agg.maxPhy[i])
            agg.maxPhy[i] = phy[i];
          agg.sumReq[i] += req[i];
          agg.sumPhy[i] += phy[i];
        }
        for (int i = 0; i < 2; ++i) {
          if (ratio[i] > agg.maxRatio[i])
            agg.maxRatio[i] = ratio[i];
          agg.sumRatio[i] += ratio[i];
        }
        if (nhits > agg.maxNHits)
          agg.maxNHits = nhits;

        snapN = agg.n;
        const double invN = 1.0 / double(snapN);
        for (int i = 0; i < 4; ++i) {
          snapMaxReq[i] = agg.maxReq[i];
          snapMeanReq[i] = agg.sumReq[i] * invN;
          snapMaxPhy[i] = agg.maxPhy[i];
        }
        for (int i = 0; i < 2; ++i) {
          snapMaxRatio[i] = agg.maxRatio[i];
          snapMeanRatio[i] = agg.sumRatio[i] * invN;
        }
        snapMaxNHits = agg.maxNHits;
      }

      const auto coutFlags = std::cout.flags();
      const auto coutPrec = std::cout.precision();

      std::cout << "============= Sizing-parameter recommendation =============" << std::endl;
      std::cout << "  Fills :  nHits=" << nhits << "  outerHits=" << outerHitsR << "  nTracksFound=" << nTracksFnd
                << "  nHitsInTracks=" << nHitsInTrk << std::endl;
      std::cout << "           nCells=" << nCellsF << "  nTriplets=" << nTripletsF << "  nCellTrackPairs=" << nCTPairsF
                << std::endl;
      std::cout << "  Caps  :  maxDoublets=" << maxDoublets << "  maxTuples=" << maxTuples << std::endl;

      // Per-event rows.
      std::cout << "  -- this event --" << std::endl;
      for (int i = 0; i < 4; ++i) {
        std::cout << "    " << std::left << std::setw(18) << paramNames[i] << " current=" << std::fixed
                  << std::setprecision(3) << std::setw(8) << currents[i] << "  phys(fill/key)=" << std::setw(8)
                  << phy[i] << "  min(fill/cap)=" << std::setw(8) << req[i] << "  recommend>= " << std::setw(8)
                  << (req[i] * kSafety) << "  headroom=" << std::setw(6)
                  << ((req[i] > 0.f) ? (currents[i] / req[i]) : 0.f) << "x" << std::endl;
      }

      // Aggregated-across-events rows
      std::cout << "  -- aggregated over " << snapN << " event" << (snapN == 1u ? "" : "s") << std::endl;
      for (int i = 0; i < 4; ++i) {
        std::cout << "    " << std::left << std::setw(18) << paramNames[i] << " current=" << std::fixed
                  << std::setprecision(3) << std::setw(8) << currents[i] << "  max(req)=" << std::setw(8)
                  << snapMaxReq[i] << "  mean(req)=" << std::setw(8) << snapMeanReq[i] << "  max(phys)=" << std::setw(8)
                  << snapMaxPhy[i] << "  recommend>= " << std::setw(8) << (snapMaxReq[i] * kSafety) << std::endl;
      }

      // Parameters that scale with nHits (doublets and tuples).
      // Safe sizing is f(nHits) = slope * nHits, where slope = max(count/nHits) across events.
      // This guarantees the observed worst-case event would have fit while
      // safety covers unobserved variation. Evaluate f at this event's nHits
      // and at the observed max nHits
      std::cout << "  -- scaling parameters f(nHits) over " << snapN << " event" << (snapN == 1u ? "" : "s") << " --"
                << std::endl;
      for (int i = 0; i < 2; ++i) {
        const double recSlope = snapMaxRatio[i] * kSafety;
        const uint64_t recAtThis = static_cast<uint64_t>(std::ceil(recSlope * double(nhits)));
        const uint64_t recAtMax = static_cast<uint64_t>(std::ceil(recSlope * double(snapMaxNHits)));
        std::cout << "    " << std::left << std::setw(20) << scaleNames[i] << " current=" << std::right << std::setw(10)
                  << scaleCurrent[i] << "  this: " << std::setw(8) << scaleCount[i] << "/" << nhits << "=" << std::fixed
                  << std::setprecision(4) << std::setw(7) << ratio[i] << "  agg max=" << std::setw(7) << snapMaxRatio[i]
                  << "  mean=" << std::setw(7) << snapMeanRatio[i] << std::endl
                  << "    " << std::setw(20) << "" << " recommend>= " << std::setprecision(4) << recSlope << " * nHits"
                  << "  (= " << recAtThis << " here,  " << recAtMax << " at observed max nHits=" << snapMaxNHits << ")"
                  << std::endl;
      }

      std::cout << "  Note: 'min'/'req' = the avg value below which the buffer would overflow." << std::endl;
      std::cout << "        Per-event 'recommend' = req * " << kSafety << " (per-event safety margin)." << std::endl;
      std::cout << "        Aggregate 'recommend' = max(req across events) * " << kSafety << std::endl;
      std::cout << "        Run a representative event sample, then read the aggregate row" << std::endl;
      std::cout << "        from the last event's report." << std::endl;
      std::cout << "===========================================================" << std::endl;

      std::cout.flags(coutFlags);
      std::cout.precision(coutPrec);
    }
#endif
    if (this->m_params.algoParams_.doStats_) {
      // counters (add flag???)

      numberOfBlocks =
          cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_doStatsForHitInTracks<TrackerTraits>{},
                          this->device_hitToTuple_->data(),
                          this->counters_->data());

      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_doStatsForTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->counters_->data());

      auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_printCounters{}, this->counters_->data());
    }
#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters: count final quality distribution, copy to host, and print funnel
    {
      // Count final track quality distribution after all processing
      auto numberOfBlocksQ = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      auto workDivQ = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocksQ, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDivQ,
                          Kernel_countFinalQuality<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          hh,
                          device_pipelineCounters_->data());
      alpaka::wait(queue);
      auto host_counters = cms::alpakatools::make_host_buffer<uint32_t[]>(caHitNtupletGenerator::kTotalCounters);
      alpaka::memcpy(queue, host_counters, *device_pipelineCounters_);
      alpaka::wait(queue);
      auto const *c = host_counters.data();
      using PC = caHitNtupletGenerator::PipelineCounter;
      printf("[CA Pipeline] Doublets: total=%u pix-pix=%u pix-OT=%u OT-OT=%u\n",
             c[PC::kDoubletsTotal],
             c[PC::kDoubletsPixPix],
             c[PC::kDoubletsPixOT],
             c[PC::kDoubletsOTOT]);
      printf("[CA Pipeline]   OT barrel: L28-29=%u(FF=%u FT=%u TT=%u) L29-30=%u(FF=%u FT=%u TT=%u)\n",
             c[PC::kDoubletsL28L29],
             c[PC::kDoubletsL28L29_FF],
             c[PC::kDoubletsL28L29_FT],
             c[PC::kDoubletsL28L29_TT],
             c[PC::kDoubletsL29L30],
             c[PC::kDoubletsL29L30_FF],
             c[PC::kDoubletsL29L30_FT],
             c[PC::kDoubletsL29L30_TT]);
      printf(
          "[CA Pipeline]            L30-31=%u(FF=%u FT=%u TT=%u) L31-32=%u(FF=%u FT=%u TT=%u) L32-33=%u(FF=%u FT=%u "
          "TT=%u)\n",
          c[PC::kDoubletsL30L31],
          c[PC::kDoubletsL30L31_FF],
          c[PC::kDoubletsL30L31_FT],
          c[PC::kDoubletsL30L31_TT],
          c[PC::kDoubletsL31L32],
          c[PC::kDoubletsL31L32_FF],
          c[PC::kDoubletsL31L32_FT],
          c[PC::kDoubletsL31L32_TT],
          c[PC::kDoubletsL32L33],
          c[PC::kDoubletsL32L33_FF],
          c[PC::kDoubletsL32L33_FT],
          c[PC::kDoubletsL32L33_TT]);
      printf("[CA Pipeline]   OT brl->disk1(z<0): L28=%u L29=%u L30=%u L31=%u L32=%u L33=%u\n",
             c[PC::kDoubletsL28D1B],
             c[PC::kDoubletsL29D1B],
             c[PC::kDoubletsL30D1B],
             c[PC::kDoubletsL31D1B],
             c[PC::kDoubletsL32D1B],
             c[PC::kDoubletsL33D1B]);
      printf("[CA Pipeline]   OT brl->disk1(z>0): L28=%u L29=%u L30=%u L31=%u L32=%u L33=%u\n",
             c[PC::kDoubletsL28D1F],
             c[PC::kDoubletsL29D1F],
             c[PC::kDoubletsL30D1F],
             c[PC::kDoubletsL31D1F],
             c[PC::kDoubletsL32D1F],
             c[PC::kDoubletsL33D1F]);
      printf("[CA Pipeline]   OT disk chain (z<0): D1-D2=%u D2-D3=%u D3-D4=%u D4-D5=%u\n",
             c[PC::kDoubletsD1BD2B],
             c[PC::kDoubletsD2BD3B],
             c[PC::kDoubletsD3BD4B],
             c[PC::kDoubletsD4BD5B]);
      printf("[CA Pipeline]   OT disk chain (z>0): D1-D2=%u D2-D3=%u D3-D4=%u D4-D5=%u other=%u\n",
             c[PC::kDoubletsD1FD2F],
             c[PC::kDoubletsD2FD3F],
             c[PC::kDoubletsD3FD4F],
             c[PC::kDoubletsD4FD5F],
             c[PC::kDoubletsOTOther]);
      // Per-cut doublet rejection counters: Total, OTEarly (L28-29), OTLate (L30-32)
      {
        using namespace caHitNtupletGenerator;
        static const char *groupNames[] = {"Total", "OTEarly(L28-29)", "OTLate(L30-32)"};
        for (int g = 0; g < 3; ++g) {
          int base = PC::kDblRejBase + g * kNCuts;
          printf(
              "[CA Pipeline] DoubletCuts %s: invalidHit=%u innerCoord=%u clusterCut=%u invalidMod=%u "
              "outerCoord=%u dzRange=%u z0=%u phi=%u zSize=%u pt=%u stubSigma=%u pixStub=%u\n",
              groupNames[g],
              c[base + kCutInvalidHit],
              c[base + kCutInnerCoord],
              c[base + kCutClusterCut],
              c[base + kCutInvalidModule],
              c[base + kCutOuterCoord],
              c[base + kCutDzRange],
              c[base + kCutZ0],
              c[base + kCutPhi],
              c[base + kCutZSize],
              c[base + kCutPt],
              c[base + kCutStubSigma],
              c[base + kCutPixStub]);
        }
      }
      printf("[CA Pipeline] Triplets: total=%u ppp=%u ppO=%u pOO=%u OOO=%u\n",
             c[PC::kTripletsTotal],
             c[PC::kTripletsPixPixPix],
             c[PC::kTripletsPixPixOT],
             c[PC::kTripletsPixOTOT],
             c[PC::kTripletsOTOTOT]);
      printf("[CA Pipeline]   OOO: barrel=%u brl->z<0=%u brl->z>0=%u z<0=%u z>0=%u other=%u\n",
             c[PC::kTripletsOOO_barrel],
             c[PC::kTripletsOOO_brlToBwd],
             c[PC::kTripletsOOO_brlToFwd],
             c[PC::kTripletsOOO_bwd],
             c[PC::kTripletsOOO_fwd],
             c[PC::kTripletsOOO_other]);
      printf("[CA Pipeline]   phiMiddle rejected: %u\n", c[PC::kTripletPhiMiddleRej]);
      printf("[CA Pipeline]   chainPhiResid rejected (early): %u\n", c[PC::kTripletChainPhiResidRej]);
      // Per-cut triplet rejection breakdown (Phase2OTStubs): how many triplet
      // candidates each TripletCuts::accept() stage rejects, in application order.
      {
        using namespace caHitNtupletGenerator;
        const int trpBase = int(kTrpRejBase);
        printf(
            "[CA Pipeline] TripletCuts rejected: alignedRZ=%u alignedXY=%u beamspotDCA=%u phiCompat=%u "
            "sameSignDPhi=%u stubGeomCurv=%u stubInnerDCurv=%u\n",
            c[trpBase + kCutAlignedRZ],
            c[trpBase + kCutAlignedXY],
            c[trpBase + kCutBeamspotCompatibleXY],
            c[trpBase + kCutPhiCompatible],
            c[trpBase + kCutSameSignDPhi],
            c[trpBase + kCutStubsCurvCompatibleWithTriplet],
            c[trpBase + kCutStubsCompatibleWithInnerDoublet]);
      }
      printf("[CA Pipeline] Fishbone killed: %u\n", c[PC::kFishboneKilled]);
      printf("[CA Pipeline] Cell status: used_in_triplet=%u killed_total=%u alive=%u\n",
             c[PC::kCellsUsedInTriplet],
             c[PC::kCellsKilledTotal],
             c[PC::kCellsAlive]);
      printf("[CA Pipeline] Cell-track pairs: %u (avgTracksPerCell=%.3f)\n",
             c[PC::kCellTrackPairs],
             c[PC::kDoubletsTotal] > 0 ? float(c[PC::kCellTrackPairs]) / float(c[PC::kDoubletsTotal]) : 0.f);
      printf("[CA Pipeline] N-tuplets: total=%u with_OT=%u with_3+OT=%u\n",
             c[PC::kNtupletsTotal],
             c[PC::kNtupletsWithOT],
             c[PC::kNtupletsOT3Plus]);
      printf("[CA Pipeline] Quality: total=%u bad=%u edup=%u dup=%u loose=%u strict=%u tight=%u HP=%u\n",
             c[PC::kQualTotal],
             c[PC::kQualBad],
             c[PC::kQualEdup],
             c[PC::kQualDup],
             c[PC::kQualLoose],
             c[PC::kQualStrict],
             c[PC::kQualTight],
             c[PC::kQualHP]);
      printf("[CA Pipeline]   with OT: strict_OT=%u tight_OT=%u HP_OT=%u\n",
             c[PC::kQualStrictWithOT],
             c[PC::kQualTightWithOT],
             c[PC::kQualHPWithOT]);
      printf("[CA Pipeline]   nhits3-4: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict34],
             c[PC::kQualTight34],
             c[PC::kQualHP34],
             c[PC::kChi2Boundary34]);
      printf("[CA Pipeline]   nhits5: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict5],
             c[PC::kQualTight5],
             c[PC::kQualHP5],
             c[PC::kChi2Boundary5]);
      printf("[CA Pipeline]   nhits6+: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict6p],
             c[PC::kQualTight6p],
             c[PC::kQualHP6p],
             c[PC::kChi2Boundary6p]);
      printf("[CA Pipeline]   fishbone: 0fb=%u 1fb=%u 2+fb=%u\n",
             c[PC::kTracksFishbone0],
             c[PC::kTracksFishbone1],
             c[PC::kTracksFishbone2p]);
      // Reset counters for next event
      alpaka::memset(queue, *device_pipelineCounters_, 0);
    }
#endif  // CA_PIPELINE_COUNTERS

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    static std::mutex lock;
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1u, 32u);
    {
      std::lock_guard<std::mutex> guard(lock);
      ++iev;
      for (uint32_t k = 0; k < 20000; k += 500) {
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_print_found_ntuplets<TrackerTraits>{},
                            hh,
                            tracks_view,
                            this->device_hitContainer_->data(),
                            this->device_hitToTuple_->data(),
                            k,
                            k + 500,
                            iev);
        alpaka::wait(queue);
      }
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_print_found_ntuplets<TrackerTraits>{},
                          hh,
                          tracks_view,
                          this->device_hitToTuple_->data(),
                          20000,
                          1000000,
                          iev);

      alpaka::wait(queue);
    }
#endif
  }

  /* This will make sense when we will be able to run this once per job in Alpaka

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::printCounters() {
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1,1);
    alpaka::exec<Acc1D>(queue_, workDiv1D, Kernel_printCounters{}, this->counters_->data());
  }
  */

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
