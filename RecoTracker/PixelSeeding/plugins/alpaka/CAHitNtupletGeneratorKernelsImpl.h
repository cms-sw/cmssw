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

  class SetHitsLayerStart {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ModulesMultiView &mm,
                                  const reco::CALayersSoAConstView &ll,
                                  uint32_t *__restrict__ hitsLayerStart) const {
      ALPAKA_ASSERT_ACC(0 == mm[0].moduleStart());

      for (int32_t i : cms::alpakatools::uniform_elements(acc, ll.metadata().size())) {
        hitsLayerStart[i] = mm[ll.layerStarts()[i]].moduleStart();
#ifdef GPU_DEBUG
        int old = i == 0 ? 0 : mm[ll.layerStarts()[i - 1]].moduleStart();
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

  class Kernel_printSizes {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
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

        // Per-layer-pair triplet cut rows. The RZ-alignment tolerance (and the stub-specific columns)
        // are anchored at the OUTER pair (L2,L3) -- its inner layer is the triplet's middle layer L2,
        // which is upstream's caThetaCut anchor. The beam-spot (DCA/floorDCA) cut is anchored at the
        // INNER pair (L1,L2) instead, whose inner layer is the triplet's innermost layer L1 --
        // upstream's caDCACut anchor. Both anchors hold on every topology. See TripletCuts::accept.
        auto tripletVectorCutsCol = tripletCuts[outerCell.layerPairId()];
        auto skips = cc[outerCell.layerPairId()].skipsLayers();

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
                                                 outerCell,
                                                 curvature,
                                                 hh,
                                                 tripletCuts,
                                                 tripletVectorCutsCol,
                                                 tripletCuts[innerCell.layerPairId()],
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
                                                        HitsMultiView hh) {
    // hasStubs enables the OT-stub-specific hit treatment (kMode filtering and the same-layer pixel
    // overlap merge). It must be the same value the fit uses, which keys off the runtime offsetStubs
    // (the start of the stub region in the merged hit collection). Derive it the same way here: a
    // Phase2OTStubs collection that carries no stubs sets offsetStubs to the unsigned sentinel
    // (-1 as int32), and there every hit is a plain pixel hit -- the binning must match the fit.
    // For non-OTStubs topologies dedupWalk applies no kMode filter and no merge (plain hit count).
    const bool hasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits> &&
                          (static_cast<int32_t>(hh.view(0).offsetStubs()) >= 0);
    return caFitHitSel::dedupWalk(foundNtuplets, it, hh, hasStubs, /*k=*/-1);
  }

  template <typename TrackerTraits>
  class Kernel_countMultiplicity {
  public:
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
          // fit chi2 and chi2Stub stay INPUTS of the network (feat[0], feat[8]). So once the DNN
          // decides a track we SKIP the stub-consistency walk entirely -- it only fed
          // maxNtupletStubChi2, whose verdict the score overwrites, so it would be wasted work. Feature
          // ORDER mirrors test/models/train_disp_nano.py FEATS (documented in CATrackDNNWeights.h).
          bool dnnHandled = false;
          if (useTrackDNN) {
            // Single-source feature fill (RecoTracker/PixelSeeding/interface/CATrackFeatures.h),
            // producing values identical to the host-side CA-features nano table producer's. On a
            // corrupt/short hit list fill() returns false -> fall through to the chi2-based path.
            float feat[caTrackFeatures::kNFeat];
            static_assert(caTrackFeatures::kNFeat == caTrackDNN::kNFeat, "feature ABI mismatch");
            const bool featOk = caTrackFeatures::fill(foundNtuplets->begin(it),
                                                      foundNtuplets->end(it),
                                                      hh,
                                                      nHitsTot,
                                                      float(tracks_view[it].nLayers()),
                                                      tracks_view[it].chi2(),
                                                      feat,
                                                      /*rzKappaOut=*/nullptr);
            // FIT-FAILURE RULE, gate half. This DNN gate REPLACED the classical `chi2 < maxChi2`
            // promotion, which rejected a failed fit as a side effect of NaN comparing false. The
            // network gives nothing for free: a non-finite input propagates through the MLP, and
            // the resulting score compared the wrong way round would promote the track. So the
            // finiteness of the network INPUTS is established BEFORE the network is evaluated --
            // never relying on a NaN surviving the sigmoid -- and a track with any non-finite
            // feature stays Quality::bad (quality() was optimistically set to strict above, so it
            // is written back explicitly). feat[0] is the fit chi2, already covered by the guard
            // at the top of the loop; this covers every other quantity the fill produced.
            bool featFinite = featOk;
            for (int k = 0; featFinite && k < int(caTrackFeatures::kNFeat); ++k)
              featFinite = !edm::isNotFinite(feat[k]);
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
              const float dnnScore = caTrackDNN_eval::score(feat);
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

          // Ntuplet-wide stub-curvature consistency: inverse-variance-weighted reduced chi2 of
          // ALL per-stub curvatures on the track around their common weighted mean (same kappa as
          // CATripletCuts). A real track's stubs agree; a combinatorial chain admitted by the
          // relaxed displaced DCA does not -> demote it below `tight`. Skipped when the DNN already
          // decided (its score subsumes this), so this is the non-DNN / fill-failed fallback;
          // CA_CHI2_DUMP forces it for the calibration printout.
#ifdef CA_CHI2_DUMP
          const bool computeStubChi2 = true;
#else
          const bool computeStubChi2 = !dnnHandled;
#endif
          if (computeStubChi2) {
            int nStubK = 0;
            float sumW = 0.f, sumWK = 0.f, sumWK2 = 0.f;
            for (auto h = foundNtuplets->begin(it); h != foundNtuplets->end(it); ++h) {
              if (*h >= static_cast<unsigned int>(nHitsTot))
                break;  // content buffer corruption from overflow
              if (!isStub(hh, *h))
                continue;  // pixel hit
              const float s = hh[*h].dPhiDrError();
              if (s > 0.f) {
                const float d = hh[*h].dPhiDr();
                const float xg = hh[*h].xGlobal();
                const float yg = hh[*h].yGlobal();
                float den, w;  // same shared kappa formula as CATrackFeatures::fill
                caTrackFeatures::stubDenWeight(xg * xg + yg * yg, d, s, den, w);
                const float k = d / std::sqrt(den);  // stub curvature
                sumW += w;
                sumWK += w * k;
                sumWK2 += w * k * k;
                ++nStubK;
              }
            }
            // chi2Stub < 0 => not enough stubs to judge consistency.
            float chi2Stub = -1.f;
            if (nStubK >= 3 && sumW > 0.f) {
              chi2Stub = (sumWK2 - sumWK * sumWK / sumW) / float(nStubK - 1);
              if (!dnnHandled && cuts.maxNtupletStubChi2 >= 0.f) {
                // Same discipline as the DNN gate: the KEEP decision is the positive comparison
                // (`chi2Stub <= cut` -> keep), so a non-finite chi2Stub -- which a degenerate stub
                // set can produce -- falls to the demoting side instead of sailing through the
                // negated `chi2Stub > cut` test. The explicit isNotFinite keeps that true under
                // -Ofast, where the comparison alone would not be trustworthy.
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

          // For Phase2OTStubs: also consider the tracks that use another stub built from the same
          // P-hit. Several stubs sharing a lowerHitIdx have different hit indices but stand for the
          // same physical measurement, so for cleaning purposes they are the same shared hit.
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            if (h < static_cast<uint32_t>(hh.size()) && isStub(hh, h)) {
              auto const lowerHitIdx = hh[h].lowerHitIdx();
              if (lowerHitIdx != std::numeric_limits<uint32_t>::max()) {
                auto const offsetStubs = hh.view(0).offsetStubs();
                auto const nHits = static_cast<uint32_t>(hh.size());
                for (uint32_t otherIdx = offsetStubs; otherIdx < nHits; ++otherIdx) {
                  if (otherIdx == h)
                    continue;
                  if (otherIdx >= hitToTuple.nOnes())
                    continue;
                  if (!isStub(hh, otherIdx))
                    continue;
                  if (hh[otherIdx].lowerHitIdx() != lowerHitIdx)
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

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
