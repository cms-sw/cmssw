#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h

// C++ headers
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

#ifdef DUMP_GPU_TK_TUPLES
#include <mutex>
#endif

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"

// local headers
#include "CACell.h"
#include "CAFishbone.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAStructures.h"

//#define CA_DEBUG
//#define CA_STATS
//#define CA_WARNINGS
//#define GPU_DEBUG
//#define NTUPLE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace caHitNtupletGeneratorKernels {

    using namespace ::caStructures;

    constexpr uint32_t tkNotFound = std::numeric_limits<uint32_t>::max();
    constexpr float maxScore = std::numeric_limits<float>::max();
    constexpr float nSigma2Phase1 = 25.f;
    constexpr float nSigma2 = 5.f;
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
          if (apc->get().first >= uint32_t(tracks_view.metadata().size()))
            printf("Tuples overflow\n");
          if (*nCells >= maxNumberOfDoublets)
            printf("Cells overflow\n");
          if (*nTrips >= uint32_t(cellCell.metadata().size()))
            printf("Triplets overflow\n");
          if (*nCellTracks >= uint32_t(cellTrack.metadata().size()))
            printf("TracksToCell overflow\n");
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
            // Mark as duplicate if both conditions are met
            const float curvi = tracks_view[it].pt();
            bool foundCompatible = false;
            // Parallelize inner loop across lanes
            for (int j = laneId; j < ntr; j += warpSize) {
              const auto jt = tracksOfCell[j];
              if (tracks_view[jt].nLayers() <= nli)
                continue;  // need a strictly longer companion
              const float dcurv = curvi - tracks_view[jt].pt();
              if (dcurv * dcurv <= 0.000001f) {
                foundCompatible = true;
                break;
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

    template <typename TrackerTraits>
    class Kernel_fastDuplicateRemover {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    CACell<TrackerTraits> const *__restrict__ cells,
                                    uint32_t const *__restrict__ nCells,
                                    CellToTrack const *__restrict__ cellTracksHisto,
                                    TkSoAView tracks_view,
                                    int32_t *__restrict__ qualityScratch,
                                    bool dupPassThrough) const {
        // quality to mark rejected
        auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
        constexpr auto loose = Quality::loose;

        ALPAKA_ASSERT_ACC(nCells);
        const auto ntNCells = (*nCells);

        auto score = [&](uint32_t it) { return tracks_view[it].chi2(); };
        auto demote = [&](uint32_t it, Quality q) {
          alpaka::atomicMin(acc, &qualityScratch[it], static_cast<int32_t>(q), alpaka::hierarchy::Blocks{});
        };

        for (auto idx : cms::alpakatools::uniform_elements(acc, ntNCells)) {
          int ntr = cellTracksHisto->size(idx);
          if (ntr < 2)
            continue;

          auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(idx);

          // Demote any track dominated by a compatible, strictly better one (higher quality, or equal
          // quality and lower chi2); each track tests all others and exact ties keep both
          for (int i = 0; i < ntr; ++i) {
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
                const auto e2dpij = nSigma2 * (iCovs[p] + tracks_view[jt].covariance()(iParam2iCov[p]));
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

          // find maxQual
          auto maxQual = reject;  // no duplicate!
          for (int i = 0; i < ntr; i++) {
            auto q = tracks_view[thisCellTracks[i]].quality();
            if (q > maxQual)
              maxQual = q;
          }

          if (maxQual <= loose)
            continue;

          // min chi2 among the best-quality tracks (read from the unmodified quality, which the dup-marking
          // above does not affect for the max-quality min-chi2 track)
          float mc = maxScore;
          for (int i = 0; i < ntr; i++) {
            auto it = thisCellTracks[i];
            if (tracks_view[it].quality() == maxQual && score(it) < mc)
              mc = score(it);
          }

          // mark all other duplicates (keep them loose)
          for (int i = 0; i < ntr; i++) {
            auto it = thisCellTracks[i];
            if (tracks_view[it].quality() > loose && score(it) > mc)
              demote(it, loose);
          }
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
                                    bool dupPassThrough) const {
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
                                    reco::CALayersSoAConstView ll,
                                    reco::CAGraphSoAConstView cc,
                                    caStructures::CAPairSoAView cn,
                                    CACell<TrackerTraits> *cells,
                                    uint32_t const *nCells,
                                    uint32_t *nTrips,
                                    HitToCell const *__restrict__ outerHitHisto,
                                    CellToCell *cellNeighborsHisto,
                                    AlgoParams const &params) const {
        using Cell = CACell<TrackerTraits>;
        uint32_t maxTriplets = cn.metadata().size();

        if (cms::alpakatools::once_per_grid(acc)) {
          *apc = 0;
        }  // ready for next kernel

        // loop on outer cells
        for (uint32_t cellIndex : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
          auto &thisCell = cells[cellIndex];
          auto innerHitId = thisCell.inner_hit_id() - hh.view(0).offsetBPIX2();

          if (int(innerHitId) < 0)
            continue;

          auto const *__restrict__ outerHitCells = outerHitHisto->begin(innerHitId);
          auto const numberOfPossibleNeighbors = outerHitHisto->size(innerHitId);

#ifdef CA_DEBUG
          printf("numberOfPossibleFromHisto;%d;%d;%d;%d;%d\n",
                 *nCells,
                 innerHitId,
                 cellIndex,
                 thisCell.innerLayer(),
                 numberOfPossibleNeighbors);
#endif
          auto ri = thisCell.inner_r(hh);
          auto zi = thisCell.inner_z(hh);
          auto ro = thisCell.outer_r(hh);
          auto zo = thisCell.outer_z(hh);
          auto thetaCut = ll[thisCell.innerLayer()].caThetaCut();
          auto skips = cc[thisCell.layerPairId()].skipsLayers();

          // loop on inner cells
          for (uint32_t j : cms::alpakatools::independent_group_elements_x(acc, numberOfPossibleNeighbors)) {
            auto otherCell = outerHitCells[j];
            auto &oc = cells[otherCell];
            auto r1 = oc.inner_r(hh);
            auto z1 = oc.inner_z(hh);
            auto dcaCut = ll[oc.innerLayer()].caDCACut();
            bool aligned = Cell::areAlignedRZ(r1, z1, ri, zi, ro, zo, params.ptmin_, thetaCut);
            if (aligned) {
              if (thisCell.dcaCut(hh, oc, dcaCut, params.hardCurvCut_)) {
                auto t_ind = alpaka::atomicAdd(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});
#ifdef CA_DEBUG
                printf("Triplet no. %d %.5f %.5f (%d %d) - %d %d -> (%d, %d, %d, %d) \n",
                       t_ind,
                       thetaCut,
                       dcaCut,
                       thisCell.layerPairId(),
                       oc.layerPairId(),
                       otherCell,
                       cellIndex,
                       thisCell.inner_hit_id(),
                       thisCell.outer_hit_id(),
                       oc.inner_hit_id(),
                       oc.outer_hit_id());
#endif

#ifdef CA_DEBUG
                printf("filling cell no. %d %d: %d -> %d\n", t_ind, cellNeighborsHisto->size(), otherCell, cellIndex);
#endif

                if (t_ind >= maxTriplets) {
#ifdef CA_WARNINGS
                  printf("Warning!!!! Too many cell->cell (triplets) associations (limit = %d)!\n",
                         cn.metadata().size());
#endif
                  alpaka::atomicSub(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});
                  break;
                }

                // One bin per cell (otherCell). The non-layer-skipping vs
                // layer-skipping distinction is encoded in bit 31 of the stored
                // outer-cell index:
                //   bit 31 = 0 -> non-layer-skipping neighbor
                //   bit 31 = 1 -> layer-skipping neighbor
                cellNeighborsHisto->count(acc, otherCell);

                cn[t_ind].inner() = otherCell;
                cn[t_ind].outer() = cellIndex | (skips ? caStructures::kSkipsLayerFlag : 0u);
                thisCell.setStatusBits(Cell::StatusBit::kUsed);
                oc.setStatusBits(Cell::StatusBit::kUsed);
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
                                    HitToCell *outerHitHisto) const {
        for (auto cellIndex : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef DOUBLETS_DEBUG
          printf("outerHitHisto;%d;%d\n", cellIndex, cells[cellIndex].outer_hit_id());
#endif
          outerHitHisto->fill(acc, cells[cellIndex].outer_hit_id() - offsetBPIX2, cellIndex);
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
        for (uint32_t index : cms::alpakatools::uniform_elements(acc, *nElements)) {
          genericHisto->fill(acc, cn[index].inner(), cn[index].outer());
        }
      }
    };

    template <typename TrackerTraits>
    class Kernel_find_ntuplets {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    HitsMultiView hh,
                                    const ::reco::CALayersSoAConstView &ll,
                                    const ::reco::CAGraphSoAConstView &cc,
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
          auto lid = thisCell.innerLayer();
          if (thisCell.inner_r() > ll[lid].startMaxInnerR())
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
              thisCell.inner_r(),
              ll[lid].startMaxInnerR());
#endif

          if (doit) {
            typename Cell::TmpTuple stack;
            typename Cell::hindex_type hits[TrackerTraits::maxHitsOnTrack];  // considering fishbone hits

            stack.reset();
            thisCell.template find_ntuplets<maxDepth>(acc,
                                                      hh,
                                                      ll,
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

    template <typename TrackerTraits>
    class Kernel_countMultiplicity {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
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
          ALPAKA_ASSERT_ACC(tracks_view[it].quality() == Quality::bad);
          if (nhits > TrackerTraits::maxHitsOnTrack)  // current limit
            printf("wrong mult %d %d\n", it, nhits);
          ALPAKA_ASSERT_ACC(nhits <= TrackerTraits::maxHitsOnTrack);
          tupleMultiplicity->count(acc, nhits);
        }
      }
    };

    template <typename TrackerTraits>
    class Kernel_fillMultiplicity {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    TkSoAView tracks_view,
                                    HitContainer const *__restrict__ foundNtuplets,
                                    TupleMultiplicity *tupleMultiplicity) const {
        for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
          auto nhits = foundNtuplets->size(it);

          if (nhits < 3)
            continue;
          if (tracks_view[it].quality() == Quality::edup)
            continue;
          ALPAKA_ASSERT_ACC(tracks_view[it].quality() == Quality::bad);
          if (nhits > TrackerTraits::maxHitsOnTrack)
            printf("wrong mult %d %d\n", it, nhits);
          ALPAKA_ASSERT_ACC(nhits <= TrackerTraits::maxHitsOnTrack);
          tupleMultiplicity->fill(acc, nhits, it);
        }
      }
    };

    template <typename TrackerTraits>
    class Kernel_classifyTracks {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    TkSoAView tracks_view,
                                    HitContainer const *__restrict__ foundNtuplets,
                                    QualityCuts<TrackerTraits> cuts) const {
        for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
          auto nhits = foundNtuplets->size(it);
          if (nhits == 0)
            break;  // guard

          // if duplicate: not even fit
          if (tracks_view[it].quality() == Quality::edup)
            continue;

          ALPAKA_ASSERT_ACC(tracks_view[it].quality() == Quality::bad);

          // mark doublets as bad
          if (nhits < 3)
            continue;

          // if the fit has any invalid parameters, mark it as bad
          bool isNaN = false;
          for (int i = 0; i < 5; ++i) {
            isNaN |= edm::isNotFinite(tracks_view[it].state()(i));
          }
          // state(2) is the (finite) inverse pt: an exactly-zero value from a straight-line or
          // numerically-degenerate fit maps to an infinite momentum in the host local-to-global
          // transform, so treat it as bad here too and never promote such a track
          isNaN |= (tracks_view[it].state()(2) == 0.f);
          if (isNaN) {
#ifdef NTUPLE_DEBUG
            printf("NaN in fit %d size %d chi2 %f\n", it, foundNtuplets->size(it), tracks_view[it].chi2());
#endif
            continue;
          }

          tracks_view[it].quality() = Quality::strict;

          if (cuts.strictCut(tracks_view, nhits, it))
            continue;

          tracks_view[it].quality() = Quality::tight;

          if (cuts.isHP(tracks_view, nhits, it))
            tracks_view[it].quality() = Quality::highPurity;
        }
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

    template <typename TrackerTraits>
    class Kernel_countHitInTracks {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    TkSoAView tracks_view,
                                    HitContainer const *__restrict__ foundNtuplets,
                                    HitToTuple *hitToTuple) const {
        for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
          if (foundNtuplets->size(idx) == 0)
            break;  // guard
          for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h)
            hitToTuple->count(acc, *h);
        }
      }
    };

    template <typename TrackerTraits>
    class Kernel_fillHitInTracks {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    TkSoAView tracks_view,
                                    HitContainer const *__restrict__ foundNtuplets,
                                    HitToTuple *hitToTuple) const {
        for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
          if (foundNtuplets->size(idx) == 0)
            break;  // guard
          for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h)
            hitToTuple->fill(acc, *h, idx);
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
                                    cms::alpakatools::AtomicPairCounter *apc) const {
        // clamp the number of tracks to the capacity of the SoA
        auto ntracks = std::min<int>(apc->get().first, tracks_view.metadata().size() - 1);
        if (cms::alpakatools::once_per_grid(acc))
          tracks_view.nTracks() = ntracks;

        // copy offsets
        for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
          tracks_view[idx].hitOffsets() = foundNtuplets->off[idx + 1];  // offset for track 0 is always 0
        }
        // fill hit indices
        for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->size())) {
          ALPAKA_ASSERT_ACC(foundNtuplets->content[idx] < static_cast<uint32_t>(hh.size()));
          track_hits_view[idx].id() = foundNtuplets->content[idx];
          track_hits_view[idx].detId() = hh[foundNtuplets->content[idx]].detectorIndex();
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
    class Kernel_fillNLayers {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    TkSoABlocksView view,
                                    uint32_t const *__restrict__ layerStarts,
                                    uint16_t maxLayers,
                                    cms::alpakatools::AtomicPairCounter *apc) const {
        // clamp the number of tracks to the capacity of the SoA
        auto ntracks = std::min<int>(apc->get().first, view.tracks().metadata().size() - 1);

        if (cms::alpakatools::once_per_grid(acc))
          view.tracks().nTracks() = ntracks;
        for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
          ALPAKA_ASSERT_ACC(reco::nHits(view.tracks(), idx) >= 3);
          view.tracks()[idx].nLayers() = reco::nLayers(view, maxLayers, layerStarts, idx);
#ifdef CA_DEBUG
          printf("Kernel_fillNLayers %d %d %d - %d %d\n",
                 idx,
                 ntracks,
                 view.tracks()[idx].nLayers(),
                 apc->get().first,
                 view.tracks().metadata().size() - 1);
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
                                    HitToTuple const *__restrict__ phitToTuple) const {
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
              const auto e2dpij = nSigma2 * (iCovs[p] + tracks_view[jt].covariance()(iParam2iCov[p]));
              if (dpij * dpij > e2dpij)
                return true;
            }
            return false;
          };

          bool dominated = false;
          for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !dominated; ++hp) {
            auto const h = *hp;
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

    // Phase-1 specialization (very forward triplets)
    template <>
    class Kernel_rejectDuplicate<pixelTopology::Phase1> {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    TkSoAView tracks_view,
                                    bool dupPassThrough,
                                    HitContainer const *__restrict__ foundNtuplets,
                                    int32_t const *__restrict__ qualityScratch,
                                    HitToTuple const *__restrict__ phitToTuple) const {
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
            if (hitToTuple.size(h) < 2)
              continue;
            int8_t maxNl = 0;
            for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
              if (qual(*jp) < longTqual)
                continue;
              maxNl = std::max(tracks_view[*jp].nLayers(), maxNl);
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

        for (auto i :
             cms::alpakatools::uniform_elements(acc, firstPrint, std::min(lastPrint, foundNtuplets->nOnes()))) {
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
            "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nLooseTracks  |  nGoodTracks | nUsedHits "
            "| "
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
      }
    };

  }  // namespace caHitNtupletGeneratorKernels

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
    uint32_t outerHits =
        nHits - offsetBPIX2;  // the number of hits that may be used as outer hits for a cell (so not on bpix1)

    // These hold the max number of associations needed
    uint32_t nHitsToTracks = std::max(uint32_t(maxTuples * algoParams.avgHitsPerTrack_), 1u);
    uint32_t nHitsToCells = std::max(uint32_t(outerHits * algoParams.avgCellsPerHit_), 1u);
    uint32_t nCellsToCells = std::max(uint32_t(maxDoublets * algoParams.avgCellsPerCell_), 1u);
    uint32_t nCellsToTracks = std::max(uint32_t(maxDoublets * algoParams.avgTracksPerCell_), 1u);

#ifdef GPU_DEBUG
    std::cout << "Allocation for tuple building with: " << std::endl;
    std::cout << "- nHits          = " << nHits << std::endl;
    std::cout << "- outerHits      = " << outerHits << std::endl;
    std::cout << "- maxDoublets    = " << maxDoublets << std::endl;
    std::cout << "- maxTracks      = " << maxTuples << std::endl;

    std::cout << "- nCellsToCells  = " << nCellsToCells << std::endl;
    std::cout << "- nHitsToCells   = " << nHitsToCells << std::endl;
    std::cout << "- nCellsToTracks = " << nCellsToTracks << std::endl;
    std::cout << "- nHitsToTracks  = " << nHitsToTracks << std::endl;
#endif

    // Hits -> Track
    device_hitToTuple_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToTupleStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToTracks);
    device_hitToTupleOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nHits + 1);
    device_hitToTupleView_ = {device_hitToTuple_->data(),
                              device_hitToTupleOffsets_->data(),
                              device_hitToTupleStorage_->data(),
                              nHits + 1,
                              nHitsToTracks};

    HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);

    // (Outer) Hits-> Cells
    device_hitToCell_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToCellStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToCells);
    device_hitToCellOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, outerHits + 1);
    device_hitToCellView_ = {device_hitToCell_->data(),
                             device_hitToCellOffsets_->data(),
                             device_hitToCellStorage_->data(),
                             outerHits + 1,
                             nHitsToCells};

    HitToCell::template launchZero<Acc1D>(device_hitToCellView_, queue);

    // Hits Phi Histograms: one histogram per layer
    device_hitPhiHist_ = cms::alpakatools::make_device_buffer<PhiBinner>(queue);
    device_phiBinnerStorage_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nHits);
    device_hitPhiView_ = {
        device_hitPhiHist_->data(), nullptr, device_phiBinnerStorage_->data(), cms::alpakatools::kDynamicSize, nHits};
    // This will hold where each layer starts in the hit soa
    device_layerStarts_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nLayers + 1);

    // Scratch quality mirror used by the (order-independent) duplicate-removal kernels
    device_qualityScratch_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, maxTuples);

    // Cell -> Neighbor Cells
    // One bin per cell (maxDoublets+1 offsets). The skipping-vs-non-skipping
    // distinction is encoded in bit 31 of each stored neighbor index:
    //   bit 31 = 0 -> non-layer-skipping neighbor
    //   bit 31 = 1 -> layer-skipping neighbor
    // This packing is possible since maxNumberOfDoublets should be well below 2^31.
    device_cellToNeighbors_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToNeighborsStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToCells);
    device_cellToNeighborsOffsets_ =
        cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, maxDoublets + 1);
    device_cellToNeighborsView_ = {device_cellToNeighbors_->data(),
                                   device_cellToNeighborsOffsets_->data(),
                                   device_cellToNeighborsStorage_->data(),
                                   maxDoublets + 1u,
                                   nCellsToCells};

    CellToCell::template launchZero<Acc1D>(device_cellToNeighborsView_, queue);

    // Cell -> Tracks
    device_cellToTracks_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToTracksStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToTracks);
    device_cellToTracksOffsets_ =
        cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, maxDoublets + 1);
    device_cellToTracksView_ = {device_cellToTracks_->data(),
                                device_cellToTracksOffsets_->data(),
                                device_cellToTracksStorage_->data(),
                                maxDoublets + 1,
                                nCellsToTracks};

    CellToTrack::template launchZero<Acc1D>(device_cellToTracksView_, queue);

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
                                maxTuples + 1,
                                nHitsToTracks};

    HitContainer::template launchZero<Acc1D>(device_hitContainerView_, queue);

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
        TrackerTraits::maxHitsOnTrack + 2,
        maxTuples};
    TupleMultiplicity::template launchZero<Acc1D>(device_tupleMultiplicityView_, queue);

    // Structures and Counters Storage
    device_simpleCells_ = cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, maxDoublets);

    device_extraStorage_ =
        cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 5u);
    device_hitTuple_apc_ = reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_extraStorage_->data());
    device_nCells_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 2));
    device_nTriplets_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 3));
    device_nCellTracks_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 4));

    deviceTriplets_ = CAPairSoACollection(queue, std::lrint(maxDoublets * algoParams.avgCellsPerCell_));
    deviceTracksCells_ = CAPairSoACollection(queue, nCellsToTracks);

    //TODO: if doStats?
    alpaka::memset(queue, *counters_, 0);

    alpaka::memset(queue, *device_nCells_, 0);
    alpaka::memset(queue, *device_nTriplets_, 0);
    alpaka::memset(queue, *device_nCellTracks_, 0);

    maxNumberOfDoublets_ = maxDoublets;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Allocations for CAHitNtupletGeneratorKernels: done!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::prepareHits(const HitsMultiView &hh,
                                                                const ModulesMultiView &mm,
                                                                const reco::CALayersSoAConstView &ll,
                                                                Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;

    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, ll.metadata().size() - 1);
    alpaka::exec<Acc1D>(queue, workDiv1D, SetHitsLayerStart{}, mm, ll, this->device_layerStarts_->data());

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
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    auto tracks_view = view.tracks();
    auto tracks_hits_view = view.trackHits();

    uint32_t nhits = static_cast<uint32_t>(hh.size());
    auto const maxDoublets = this->maxNumberOfDoublets_;
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
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    const auto rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
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
                        ll,
                        cc,
                        this->deviceTriplets_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_hitToCell_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->m_params.algoParams_);

    CellToCell::template launchFinalize<Acc1D>(this->device_cellToNeighborsView_, queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_connect -> Done!" << std::endl;
#endif

    auto threadsPerBlock = 1024;
    auto blocks = cms::alpakatools::divide_up_by(std::lrint(maxDoublets * m_params.algoParams_.avgCellsPerCell_),
                                                 threadsPerBlock);
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
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_hitToCell_->data(),
                          this->device_cellToTracks_->data(),
                          nhits - offsetBPIX2,
                          false,
                          this->m_params.algoParams_.onlySameLayersFishbone_);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Early fishbone -> Done!" << std::endl;
#endif
    }
    blockSize = 64;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_find_ntuplets<TrackerTraits>{},
                        hh,
                        ll,
                        cc,
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

    CellToTracks::template launchFinalize<Acc1D>(this->device_cellToTracksView_, queue);

    blocks = cms::alpakatools::divide_up_by(std::lrint(maxDoublets * m_params.algoParams_.avgCellsPerCell_),
                                            threadsPerBlock);
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

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillHitDetIndices<TrackerTraits>{},
                        tracks_view,
                        tracks_hits_view,
                        this->device_hitContainer_->data(),
                        hh,
                        this->device_hitTuple_apc_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillHitDetIndices   -> done!" << std::endl;
#endif

    if (this->m_params.algoParams_.doEarlyDuplicateRemover_) {
      if constexpr (std::is_same_v<pixelTopology::Phase1, TrackerTraits>) {
        // for Phase-1, the workdivision is simpler and therefore needs a separate call here
        // remove duplicates (tracks that share a doublet)
        numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
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
        // remove duplicates (tracks that share a doublet).
        // Warp-cooperative work division: one cell per Y-thread, one warp per cell.
        const uint32_t stride = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
        const uint32_t cellsPerBlock = 8;  // 8 cells * 32 threads = 256 threads/block
        // gridDim.y is capped at 65535 by CUDA. If maxDoublets needs more, uniform_elements_y
        // strides each Y-thread across the remaining cells.
        auto earlyDupRemoverNumBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, cellsPerBlock);
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
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_countMultiplicity<TrackerTraits>{},
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
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_hitToCell_->data(),
                          this->device_cellToTracks_->data(),
                          nhits - offsetBPIX2,
                          true,
                          this->m_params.algoParams_.onlySameLayersFishbone_);
    }

#ifdef GPU_DEBUG
    std::cout << "lateFishbone -> done!" << std::endl;
    alpaka::wait(queue);
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::buildDoublets(const HitsMultiView &hh,
                                                                  const ::reco::CAGraphSoAConstView &cc,
                                                                  const ::reco::CALayersSoAConstView &ll,
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
      return;  // protect against empty events

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
    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        GetDoubletsFromHisto<TrackerTraits>{},
                        maxDoublets,
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        hh,
                        cc,
                        ll,
                        this->device_layerStarts_->data(),
                        this->device_hitPhiHist_->data(),
                        this->device_hitToCell_->data(),
                        this->m_params.algoParams_);

    HitToCell::template launchFinalize<Acc1D>(this->device_hitToCellView_, queue);

    threadsPerBlock = 512;
    blocks = cms::alpakatools::divide_up_by(maxDoublets, threadsPerBlock);
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
                        this->device_hitToCell_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "FillDoubletsHisto   -> done!" << std::endl;
#endif
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

    uint32_t nhits = hh.size();

    auto blockSize = 64;
    auto const maxDoublets = this->maxNumberOfDoublets_;
    auto const maxTuples = tracks_view.metadata().size();

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
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_classifyTracks<TrackerTraits>{},
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->m_params.qualityCuts_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_classifyTracks -> done!" << std::endl;
#endif

    if (this->m_params.algoParams_.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
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
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
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
                          this->m_params.algoParams_.dupPassThrough_);
      applyQuality();
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_fastDuplicateRemover   -> done!" << std::endl;
#endif
    }
    if (this->m_params.algoParams_.doSharedHitCut_ || this->m_params.algoParams_.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data());

      GenericContainer::template launchFinalize<Acc1D>(this->device_hitToTupleView_, queue);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_countHitInTracks   -> done!" << std::endl;
#endif
    }

    if (this->m_params.algoParams_.doSharedHitCut_) {
      // mark duplicates (tracks that share at least one hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
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
                          this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_rejectDuplicate   -> done!" << std::endl;
#endif

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
      numberOfBlocks = cms::alpakatools::divide_up_by(std::max(nhits, maxDoublets), blockSize);
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

#ifdef CA_STATS
    alpaka::wait(queue);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_printSizes{},
                        hh,
                        tracks_view,
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_nCellTracks_->data());

    alpaka::wait(queue);
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

      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
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
