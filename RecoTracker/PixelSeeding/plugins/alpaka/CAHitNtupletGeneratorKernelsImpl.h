#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS

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
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"

// local includes
#include "CACell.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels {

  using namespace ::caStructures;

  constexpr uint32_t tkNotFound = std::numeric_limits<uint32_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
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
                                  const reco::HitModuleSoAConstView &mm,
                                  const reco::CALayersSoAConstView &ll,
                                  uint32_t *__restrict__ hitsLayerStart) const {
      ALPAKA_ASSERT_ACC(0 == mm.moduleStart()[0]);

      for (int32_t i : cms::alpakatools::uniform_elements(acc, ll.metadata().size())) {
        hitsLayerStart[i] = mm.moduleStart()[ll.layerStarts()[i]];
#ifdef GPU_DEBUG
        int old = i == 0 ? 0 : mm.moduleStart()[ll.layerStarts()[i - 1]];
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
                                  HitsConstView hh,
                                  TkSoAView tt,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks) const {
      if (cms::alpakatools::once_per_grid(acc))
        printf(
            "nSizes: hh.metadata().size() %d; hh.metadata().size() - hh.offsetBPIX2() %d; nCells %d; nTrips %d; "
            "nCellTracks %d; nTracks %d; tt.metadata().size() %d\n",
            hh.metadata().size(),
            hh.metadata().size() - hh.offsetBPIX2(),
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

  // assume the above (so, short tracks already removed)
  template <typename TrackerTraits>
  class Kernel_fastDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      constexpr auto loose = Quality::loose;

      ALPAKA_ASSERT_ACC(nCells);
      const auto ntNCells = (*nCells);

      for (auto idx : cms::alpakatools::uniform_elements(acc, ntNCells)) {
        if (cellTracksHisto->size(idx) < 2)
          continue;

        float mc = maxScore;
        uint32_t im = tkNotFound;

        // auto score = [&](auto it) { return std::abs(reco::tip(tracks_view, it)); };
        auto score = [&](auto it) { return tracks_view[it].chi2(); };

        // full crazy combinatorics
        auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(idx);
        int ntr = cellTracksHisto->size(idx);
        for (int i = 0; i < ntr - 1; i++) {
          auto it = thisCellTracks[i];
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;

          // get track parameters and covariances
          float iParams[nTrackParameters];
          float iCovs[nTrackParameters];
          for (int p{0}; p < nTrackParameters; ++p) {
            iParams[p] = tracks_view[it].state()(p);
            const auto c = iParam2iCov[p];
            iCovs[p] = tracks_view[it].covariance()(c);
          }
          // function that compares the five track parameters of tracks it and jt
          auto incompatibleTrackParams = [=](int jt) -> bool {
            // comparing phi, tip, 1/pT, cotan(theta) and zip
            for (int p{0}; p < nTrackParameters; ++p) {
              const auto dpij = iParams[p] - tracks_view[jt].state()(p);
              const auto c = iParam2iCov[p];
              const auto e2dpij = nSigma2 * (iCovs[p] + tracks_view[jt].covariance()(c));
              if (dpij * dpij > e2dpij)
                return true;  // incompatible param found
            }
            return false;  // all params compatible
          };

          // loop over remaining tracks j and compare
          for (int j = i + 1; j < ntr; ++j) {
            auto jt = thisCellTracks[j];
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            if (incompatibleTrackParams(jt))
              continue;
            if ((qj < qi) || (qj == qi && score(it) < score(jt)))
              tracks_view[jt].quality() = reject;
            // explicitly check since they might be identical when using the same hits for fitting!
            else if ((qj > qi) || (qj == qi && score(it) > score(jt))) { 
              tracks_view[it].quality() = reject;
              break;
            }
          }
        }

        // find maxQual
        auto maxQual = reject;  // no duplicate!
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() > maxQual)
            maxQual = tracks_view[it].quality();
        }

        if (maxQual <= loose)
          continue;

        // find min score
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() == maxQual && score(it) < mc) {
            mc = score(it);
            im = it;
          }
        }

        if (tkNotFound == im)
          continue;

        // mark all other duplicates  (not yet, keep it loose)
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() > loose && score(it) > mc)
            tracks_view[it].quality() = loose;  //no race:  simple assignment of the same constant
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_connect {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const &acc,
                                  cms::alpakatools::AtomicPairCounter *apc,  // just to zero them
                                  HitsConstView hh,
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
        auto innerHitId = thisCell.inner_hit_id() - hh.offsetBPIX2();

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
                printf("Warning!!!! Too many cell->cell (triplets) associations (limit = %d)!\n", cn.metadata().size());
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
                                  HitsConstView hh,
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
                                  HitsConstView hh,
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
        ALPAKA_ASSERT_ACC(foundNtuplets->content[idx] < (uint32_t)hh.metadata().size());
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

  // mostly for very forward triplets.....
  template <typename TrackerTraits>
  class Kernel_rejectDuplicate {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        // auto score = [&](auto it, auto nl) { return std::abs(reco::tip(tracks_view, it)); };
        auto score = [&](auto it, auto nl) { return tracks_view[it].chi2(); };

        // full combinatorics
        for (auto ip = hitToTuple.begin(idx); ip < hitToTuple.end(idx) - 1; ++ip) {
          auto const it = *ip;
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;

          // get track parameters and covariances
          float iParams[nTrackParameters];
          float iCovs[nTrackParameters];
          for (int p{0}; p < nTrackParameters; ++p) {
            iParams[p] = tracks_view[it].state()(p);
            const auto c = iParam2iCov[p];
            iCovs[p] = tracks_view[it].covariance()(c);
          }
          // function that compares the five track parameters of tracks it and jt
          auto incompatibleTrackParams = [=](int jt) -> bool {
            // comparing phi, tip, 1/pT, cotan(theta) and zip
            for (int p{0}; p < nTrackParameters; ++p) {
              const auto dpij = iParams[p] - tracks_view[jt].state()(p);
              const auto c = iParam2iCov[p];
              const auto e2dpij = nSigma2 * (iCovs[p] + tracks_view[jt].covariance()(c));
              if (dpij * dpij > e2dpij)
                return true;  // incompatible param found
            }
            return false;  // all params compatible
          };

          auto nli = tracks_view[it].nLayers();

          for (auto jp = ip + 1; jp < hitToTuple.end(idx); ++jp) {
            auto const jt = *jp;
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            if (incompatibleTrackParams(jt))
              continue;
            auto nlj = tracks_view[jt].nLayers();
            if (nlj < nli || (nlj == nli && (qj < qi || (qj == qi && score(it, nli) < score(jt, nlj)))))
              tracks_view[jt].quality() = reject;
            // explicitly check since we can have actual duplicated tracks with identical parameters
            else if (nli < nlj || (nli == nlj && (qi < qj || (qi == qj && score(jt, nlj) < score(it, nli))))){
              tracks_view[it].quality() = reject;
              break;
            }
            // if we have two tracks with the same length, parameters and quality, we keep the one with the lower index 
            // (arbitrary but deterministic) and reject the other to avoid double counting
            else if (it < jt)
              tracks_view[jt].quality() = reject;
            else
              tracks_view[it].quality() = reject;
          }
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_sharedHitCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
                                  uint32_t const *__restrict__ layerStarts,
                                  TkSoAView tracks_view,
                                  int nmin,
                                  bool dupPassThrough,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      // quality of longest track
      auto const longTqual = Quality::highPurity;

      auto &hitToTuple = *phitToTuple;

      uint32_t l1end = layerStarts[1];

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        // checking if shared hit is on bpix1
        if (idx < l1end)
          continue;

        int8_t maxNl = 0;

        // find maxNl
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (tracks_view[*it].quality() < longTqual)
            continue;
          // if (tracks_view[*it].nHits()==3) continue;
          auto nl = tracks_view[*it].nLayers();
          maxNl = std::max(nl, maxNl);
        }

        if (maxNl < 4)
          continue;

        // quad pass through (leave for tests)
        // maxNl = std::min(4, maxNl);

        // kill all tracks shorter than maxHl (only triplets???
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          auto nl = tracks_view[*it].nLayers();

          // checking if the tuple is short enough
          if (nl > nmin)
            continue;

          if (nl < maxNl && tracks_view[*it].quality() > reject)
            tracks_view[*it].quality() = reject;
        }
      }
    }
  };
  template <typename TrackerTraits>
  class Kernel_tripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::strict;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        float mc = maxScore;
        uint32_t im = tkNotFound;
        bool onlyTriplets = true;

        // check if only triplets
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (tracks_view[*it].quality() <= good)
            continue;
          onlyTriplets &= reco::isTriplet(tracks_view, *it);
          if (!onlyTriplets)
            break;
        }

        // only triplets
        if (!onlyTriplets)
          continue;

        // for triplets choose best tip!  (should we first find best quality???)
        for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
          auto const it = *ip;
          if (tracks_view[it].quality() >= good && std::abs(reco::tip(tracks_view, it)) < mc) {
            mc = std::abs(reco::tip(tracks_view, it));
            im = it;
          }
        }

        if (tkNotFound == im)
          continue;

        // mark worse ambiguities
        for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
          auto const it = *ip;
          if (tracks_view[it].quality() > reject && it != im)
            tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
        }

      }  // loop over hits
    }
  };

  template <typename TrackerTraits>
  class Kernel_simpleTripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::loose;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        float mc = maxScore;
        uint32_t im = tkNotFound;

        // choose best tip!  (should we first find best quality???)
        for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
          auto const it = *ip;
          if (tracks_view[it].quality() >= good && std::abs(reco::tip(tracks_view, it)) < mc) {
            mc = std::abs(reco::tip(tracks_view, it));
            im = it;
          }
        }

        if (tkNotFound == im)
          continue;

        // mark worse ambiguities
        for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
          auto const it = *ip;
          if (tracks_view[it].quality() > reject && reco::isTriplet(tracks_view, it) && it != im)
            tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
        }

      }  // loop over hits
    }
  };

  template <typename TrackerTraits>
  class Kernel_print_found_ntuplets {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
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
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
