#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h

//#define GPU_DEBUG
//#define NTUPLE_DEBUG

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

// local includes
#include "CACell.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels {

  constexpr uint32_t tkNotFound = std::numeric_limits<uint16_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  constexpr float nSigma2 = 25.f;

  // all of these below are mostly to avoid brining around the relative namespace

  template <typename TrackerTraits>
  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;

  template <typename TrackerTraits>
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

  template <typename TrackerTraits>
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;

  template <typename TrackerTraits>
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;

  template <typename TrackerTraits>
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

  using Quality = ::pixelTrack::Quality;

  template <typename TrackerTraits>
  using TkSoAView = reco::TrackSoAView<TrackerTraits>;

  template <typename TrackerTraits>
  using HitContainer = typename reco::TrackSoA<TrackerTraits>::HitContainer;

  template <typename TrackerTraits>
  using HitsConstView = typename CACellT<TrackerTraits>::HitsConstView;

  template <typename TrackerTraits>
  using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;

  template <typename TrackerTraits>
  using CAParams = caHitNtupletGenerator::CAParamsT<TrackerTraits>;

  using Counters = caHitNtupletGenerator::Counters;

  template <typename TrackerTraits>
  class Kernel_checkOverflows {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  TupleMultiplicity<TrackerTraits> const *tupleMultiplicity,
                                  HitToTuple<TrackerTraits> const *hitToTuple,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  CACellT<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellNeighborsVector<TrackerTraits> const *cellNeighbors,
                                  CellTracksVector<TrackerTraits> const *cellTracks,
                                  OuterHitOfCell<TrackerTraits> const *isOuterHitOfCell,
                                  int32_t nHits,
                                  uint32_t maxNumberOfDoublets,
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
        if (apc->get().first < TrackerTraits::maxNumberOfQuadruplets) {
          ALPAKA_ASSERT_ACC(tracks_view.hitIndices().size(apc->get().first) == 0);
          ALPAKA_ASSERT_ACC(tracks_view.hitIndices().size() == apc->get().second);
        }
      }

      for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().nOnes())) {
        if (tracks_view.hitIndices().size(idx) > TrackerTraits::maxHitsOnTrack)  // current real limit
          printf("ERROR %d, %d\n", idx, tracks_view.hitIndices().size(idx));
        ALPAKA_ASSERT_ACC(ftracks_view.hitIndices().size(idx) <= TrackerTraits::maxHitsOnTrack);
        for (auto ih = tracks_view.hitIndices().begin(idx); ih != tracks_view.hitIndices().end(idx); ++ih)
          ALPAKA_ASSERT_ACC(int(*ih) < nHits);
      }
#endif

      if (cms::alpakatools::once_per_grid(acc)) {
#ifdef GPU_DEBUG
        if (apc->get().first >= TrackerTraits::maxNumberOfQuadruplets)
          printf("Tuples overflow\n");
        if (*nCells >= maxNumberOfDoublets)
          printf("Cells overflow\n");
        if (cellNeighbors && cellNeighbors->full())
          printf("cellNeighbors overflow %d %d \n", cellNeighbors->capacity(), cellNeighbors->size());
        if (cellTracks && cellTracks->full())
          printf("cellTracks overflow\n");
        if (int(hitToTuple->nOnes()) < nHits)
          printf("ERROR hitToTuple  overflow %d %d\n", hitToTuple->nOnes(), nHits);
        printf("size of cellNeighbors %d \n cellTracks %d \n hitToTuple %d \n",
               cellNeighbors->size(),
               cellTracks->size(),
               hitToTuple->size());
#endif
      }

      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];
        if (thisCell.hasFishbone() && !thisCell.isKilled())
          alpaka::atomicAdd(acc, &c.nFishCells, 1ull, alpaka::hierarchy::Blocks{});
        if (thisCell.outerNeighbors().full())  //++tooManyNeighbors[thisCell.theLayerPairId];
          printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.layerPairId());
        if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
          printf("Tracks overflow %d in %d\n", idx, thisCell.layerPairId());
        if (thisCell.isKilled())
          alpaka::atomicAdd(acc, &c.nKilledCells, 1ull, alpaka::hierarchy::Blocks{});
        if (!thisCell.unused())
          alpaka::atomicAdd(acc, &c.nEmptyCells, 1ull, alpaka::hierarchy::Blocks{});
        if ((0 == hitToTuple->size(thisCell.inner_hit_id())) && (0 == hitToTuple->size(thisCell.outer_hit_id())))
          alpaka::atomicAdd(acc, &c.nZeroTrackCells, 1ull, alpaka::hierarchy::Blocks{});
      }

      // FIXME this loop was up to nHits - isOuterHitOfCell.offset in the CUDA version
      for (auto idx : cms::alpakatools::uniform_elements(acc, nHits))
        if ((*isOuterHitOfCell).container[idx].full())  // ++tooManyOuterHitOfCell;
          printf("OuterHitOfCell overflow %d\n", idx);
    }
  };

  template <typename TrackerTraits>
  class Kernel_fishboneCleaner {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  CACellT<TrackerTraits> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  TkSoAView<TrackerTraits> tracks_view) const {
      constexpr auto reject = Quality::dup;

      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];
        if (!thisCell.isKilled())
          continue;

        for (auto it : thisCell.tracks())
          tracks_view[it].quality() = reject;
      }
    }
  };

  // remove shorter tracks if sharing a cell
  // It does not seem to affect efficiency in any way!
  template <typename TrackerTraits>
  class Kernel_earlyDuplicateRemover {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  CACellT<TrackerTraits> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      constexpr auto reject = pixelTrack::Quality::edup;  /// cannot be loose
      ALPAKA_ASSERT_ACC(nCells);
      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];

        if (thisCell.tracks().size() < 2)
          continue;

        int8_t maxNl = 0;

        // find maxNl
        for (auto it : thisCell.tracks()) {
          auto nl = tracks_view[it].nLayers();
          maxNl = std::max(nl, maxNl);
        }

        // if (maxNl<4) continue;
        // quad pass through (leave it here for tests)
        //  maxNl = std::min(4, maxNl);

        for (auto it : thisCell.tracks()) {
          if (tracks_view[it].nLayers() < maxNl)
            tracks_view[it].quality() = reject;  // no race: simple assignment of the same constant
        }
      }
    }
  };

  // assume the above (so, short tracks already removed)
  template <typename TrackerTraits>
  class Kernel_fastDuplicateRemover {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  CACellT<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      constexpr auto loose = Quality::loose;

      ALPAKA_ASSERT_ACC(nCells);
      const auto ntNCells = (*nCells);

      for (auto idx : cms::alpakatools::uniform_elements(acc, ntNCells)) {
        auto const &thisCell = cells[idx];
        if (thisCell.tracks().size() < 2)
          continue;

        float mc = maxScore;
        uint16_t im = tkNotFound;

        auto score = [&](auto it) { return std::abs(reco::tip(tracks_view, it)); };

        // full crazy combinatorics
        int ntr = thisCell.tracks().size();
        for (int i = 0; i < ntr - 1; ++i) {
          auto it = thisCell.tracks()[i];
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;
          auto opi = tracks_view[it].state()(2);
          auto e2opi = tracks_view[it].covariance()(9);
          auto cti = tracks_view[it].state()(3);
          auto e2cti = tracks_view[it].covariance()(12);
          for (auto j = i + 1; j < ntr; ++j) {
            auto jt = thisCell.tracks()[j];
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            auto opj = tracks_view[jt].state()(2);
            auto ctj = tracks_view[jt].state()(3);
            auto dct = nSigma2 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto dop = nSigma2 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            if ((qj < qi) || (qj == qi && score(it) < score(jt)))
              tracks_view[jt].quality() = reject;
            else {
              tracks_view[it].quality() = reject;
              break;
            }
          }
        }

        // find maxQual
        auto maxQual = reject;  // no duplicate!
        for (auto it : thisCell.tracks()) {
          if (tracks_view[it].quality() > maxQual)
            maxQual = tracks_view[it].quality();
        }

        if (maxQual <= loose)
          continue;

        // find min score
        for (auto it : thisCell.tracks()) {
          if (tracks_view[it].quality() == maxQual && score(it) < mc) {
            mc = score(it);
            im = it;
          }
        }

        if (tkNotFound == im)
          continue;

        // mark all other duplicates  (not yet, keep it loose)
        for (auto it : thisCell.tracks()) {
          if (tracks_view[it].quality() > loose && it != im)
            tracks_view[it].quality() = loose;  //no race:  simple assignment of the same constant
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_connect {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  cms::alpakatools::AtomicPairCounter *apc1,
                                  cms::alpakatools::AtomicPairCounter *apc2,  // just to zero them
                                  HitsConstView<TrackerTraits> hh,
                                  CACellT<TrackerTraits> *cells,
                                  uint32_t *nCells,
                                  CellNeighborsVector<TrackerTraits> *cellNeighbors,
                                  OuterHitOfCell<TrackerTraits> const *isOuterHitOfCell,
                                  CAParams<TrackerTraits> params) const {
      using Cell = CACellT<TrackerTraits>;

      if (cms::alpakatools::once_per_grid(acc)) {
        *apc1 = 0;
        *apc2 = 0;
      }  // ready for next kernel

      // loop on outer cells
      for (uint32_t idx : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
        auto cellIndex = idx;
        auto &thisCell = cells[idx];
        auto innerHitId = thisCell.inner_hit_id();
        if (int(innerHitId) < isOuterHitOfCell->offset)
          continue;

        uint32_t numberOfPossibleNeighbors = (*isOuterHitOfCell)[innerHitId].size();
        auto vi = (*isOuterHitOfCell)[innerHitId].data();
        auto ri = thisCell.inner_r(hh);
        auto zi = thisCell.inner_z(hh);
        auto ro = thisCell.outer_r(hh);
        auto zo = thisCell.outer_z(hh);
        float caThetaCut;
        auto isOuterBarrelPixel = thisCell.outer_detIndex(hh) < TrackerTraits::last_barrel_detIndex;
        auto isInnerBarrelPixel = thisCell.inner_detIndex(hh) < TrackerTraits::last_barrel_detIndex;
        auto isOuterForwardPixel = thisCell.outer_detIndex(hh) >= TrackerTraits::last_barrel_detIndex &&
                                   thisCell.outer_detIndex(hh) < TrackerTraits::numberOfPixelModules;
        auto isOuterBarrelStrip =
            thisCell.outer_detIndex(hh) >= TrackerTraits::numberOfPixelModules && thisCell.outer_detIndex(hh) < 3392;
        auto isInnerBarrelStrip =
            thisCell.inner_detIndex(hh) >= TrackerTraits::numberOfPixelModules && thisCell.inner_detIndex(hh) < 3392;
        auto isOuterForwardStrip = thisCell.outer_detIndex(hh) >= 3392;
        caThetaCut = (isInnerBarrelPixel && isOuterBarrelPixel)    ? params.CAThetaCutBarrel_
                     : (isInnerBarrelPixel && isOuterForwardPixel) ? params.CAThetaCutForward_
                     : (isInnerBarrelPixel && isOuterBarrelStrip)  ? params.CAThetaCutBarrelPixelBarrelStrip_
                     : (isInnerBarrelPixel && isOuterForwardStrip) ? params.CAThetaCutBarrelPixelForwardStrip_
                     : (isInnerBarrelStrip && isOuterForwardStrip) ? params.CAThetaCutBarrelStripForwardStrip_
                     : (isInnerBarrelStrip && isOuterBarrelStrip)  ? params.CAThetaCutBarrelStrip_
                                                                   : params.CAThetaCutDefault_;
        // loop on inner cells
        for (uint32_t j : cms::alpakatools::independent_group_elements_x(acc, numberOfPossibleNeighbors)) {
          auto otherCell = (vi[j]);
          auto &oc = cells[otherCell];
          auto r1 = oc.inner_r(hh);
          auto z1 = oc.inner_z(hh);
          bool aligned = Cell::areAlignedRZ(r1,
                                            z1,
                                            ri,
                                            zi,
                                            ro,
                                            zo,
                                            params.ptmin_,
                                            caThetaCut);  // 2.f*thetaCut); // FIXME tune cuts
          auto isOuterBarrelPixel = oc.outer_detIndex(hh) < TrackerTraits::last_barrel_detIndex;
          auto isOuterForwardPixel = oc.outer_detIndex(hh) >= TrackerTraits::last_barrel_detIndex &&
                                     oc.outer_detIndex(hh) < TrackerTraits::numberOfPixelModules;
          auto isOuterBarrelStrip =
              oc.outer_detIndex(hh) >= TrackerTraits::numberOfPixelModules && oc.outer_detIndex(hh) < 3392;
          auto isInnerBarrelStrip =
              oc.inner_detIndex(hh) >= TrackerTraits::numberOfPixelModules && oc.inner_detIndex(hh) < 3392;
          auto isOuterForwardStrip = oc.outer_detIndex(hh) >= 3392;
          auto isInnerForwardStrip = oc.inner_detIndex(hh) >= 3392;
          auto isFirstInnerBarrelPixel = oc.inner_detIndex(hh) < TrackerTraits::last_bpix1_detIndex;
          auto isBeyondFirstInnerBarrelPixel = oc.inner_detIndex(hh) > TrackerTraits::last_bpix1_detIndex &&
                                               oc.inner_detIndex(hh) < TrackerTraits::numberOfPixelModules;
          float dcaCutTriplet;
          dcaCutTriplet = (isFirstInnerBarrelPixel && (isOuterBarrelStrip || isOuterForwardStrip))
                              ? params.dcaCutInnerTripletPixelStrip_
                          : (isBeyondFirstInnerBarrelPixel && (isOuterBarrelStrip || isOuterForwardStrip))
                              ? params.dcaCutOuterTripletPixelStrip_
                          : (isFirstInnerBarrelPixel && (isOuterBarrelPixel || isOuterForwardPixel))
                              ? params.dcaCutInnerTriplet_
                          : (isBeyondFirstInnerBarrelPixel && (isOuterBarrelPixel || isOuterForwardPixel))
                              ? params.dcaCutOuterTriplet_
                          : ((isInnerBarrelStrip || isInnerForwardStrip) && (isOuterBarrelStrip || isOuterForwardStrip))
                              ? params.dcaCutTripletStrip_
                              : params.dcaCutTripletDefault_;
          if (aligned && thisCell.dcaCut(hh, oc, dcaCutTriplet,
                                         params.hardCurvCut_)) {  // FIXME tune cuts
            oc.addOuterNeighbor(acc, cellIndex, *cellNeighbors);
            thisCell.setStatusBits(Cell::StatusBit::kUsed);
            oc.setStatusBits(Cell::StatusBit::kUsed);
          }
        }  // loop on inner cells
      }  // loop on outer cells
    }
  };
  template <typename TrackerTraits>
  class Kernel_find_ntuplets {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  HitsConstView<TrackerTraits> hh,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  CACellT<TrackerTraits> *__restrict__ cells,
                                  uint32_t const *nCells,
                                  CellTracksVector<TrackerTraits> *cellTracks,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  CAParams<TrackerTraits> params) const {
      // recursive: not obvious to widen

      using Cell = CACellT<TrackerTraits>;
#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc))
        printf("starting producing ntuplets from %d cells \n", *nCells);
#endif
      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto const &thisCell = cells[idx];

        // cut by earlyFishbone
        if (thisCell.isKilled())
          continue;

        // we require at least three hits
        if (thisCell.outerNeighbors().empty())
          continue;

        auto pid = thisCell.layerPairId();
        bool doit = params.startingLayerPair(pid);

        constexpr uint32_t maxDepth = TrackerTraits::maxDepth;

        if (doit) {
          typename Cell::TmpTuple stack;
          stack.reset();
          bool bpix1Start = params.startAt0(pid);
          thisCell.template find_ntuplets<maxDepth, TAcc>(acc,
                                                          hh,
                                                          cells,
                                                          *cellTracks,
                                                          tracks_view.hitIndices(),
                                                          *apc,
                                                          tracks_view.quality(),
                                                          stack,
                                                          params.minHitsPerNtuplet_,
                                                          bpix1Start);
          ALPAKA_ASSERT_ACC(stack.empty());
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_mark_used {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  CACellT<TrackerTraits> *__restrict__ cells,
                                  uint32_t const *nCells) const {
      using Cell = CACellT<TrackerTraits>;
      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto &thisCell = cells[idx];
        if (!thisCell.tracks().empty())
          thisCell.setStatusBits(Cell::StatusBit::kInTrack);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_countMultiplicity {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  TupleMultiplicity<TrackerTraits> *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().nOnes())) {
        auto nhits = tracks_view.hitIndices().size(it);
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
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  TupleMultiplicity<TrackerTraits> *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().nOnes())) {
        auto nhits = tracks_view.hitIndices().size(it);
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
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  QualityCuts<TrackerTraits> cuts) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().nOnes())) {
        auto nhits = tracks_view.hitIndices().size(it);
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
        if (isNaN) {
#ifdef NTUPLE_DEBUG
          printf("NaN in fit %d size %d chi2 %f\n", it, tracks_view.hitIndices().size(it), tracks_view[it].chi2());
#endif
          continue;
        }

        tracks_view[it].quality() = Quality::strict;

        if (cuts.strictCut(tracks_view, it))
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
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc, TkSoAView<TrackerTraits> tracks_view, Counters *counters) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().nOnes())) {
        if (tracks_view.hitIndices().size(idx) == 0)
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
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  HitToTuple<TrackerTraits> *hitToTuple) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().nOnes())) {
        if (tracks_view.hitIndices().size(idx) == 0)
          break;  // guard
        for (auto h = tracks_view.hitIndices().begin(idx); h != tracks_view.hitIndices().end(idx); ++h)
          hitToTuple->count(acc, *h);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitInTracks {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  HitToTuple<TrackerTraits> *hitToTuple) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().nOnes())) {
        if (tracks_view.hitIndices().size(idx) == 0)
          break;  // guard
        for (auto h = tracks_view.hitIndices().begin(idx); h != tracks_view.hitIndices().end(idx); ++h)
          hitToTuple->fill(acc, *h, idx);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitDetIndices {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  HitsConstView<TrackerTraits> hh) const {
      // copy offsets
      for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().nOnes())) {
        tracks_view.detIndices().off[idx] = tracks_view.hitIndices().off[idx];
      }
      // fill hit indices
      for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.hitIndices().size())) {
        ALPAKA_ASSERT_ACC(tracks_view.hitIndices().content[idx] < (uint32_t)hh.metadata().size());
        tracks_view.detIndices().content[idx] = hh[tracks_view.hitIndices().content[idx]].detectorIndex();
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillNLayers {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  cms::alpakatools::AtomicPairCounter *apc) const {
      // clamp the number of tracks to the capacity of the SoA
      auto ntracks = std::min<int>(apc->get().first, tracks_view.metadata().size() - 1);

      if (cms::alpakatools::once_per_grid(acc))
        tracks_view.nTracks() = ntracks;
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        ALPAKA_ASSERT_ACC(TracksUtilities<TrackerTraits>::nHits(tracks_view, idx) >= 3);
        tracks_view[idx].nLayers() = TracksUtilities<TrackerTraits>::computeNumberOfLayers(tracks_view, idx);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_doStatsForHitInTracks {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  HitToTuple<TrackerTraits> const *__restrict__ hitToTuple,
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
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  int *__restrict__ nshared,
                                  HitContainer<TrackerTraits> const *__restrict__ ptuples,
                                  Quality const *__restrict__ quality,
                                  HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) const {
      constexpr auto loose = Quality::loose;

      auto &hitToTuple = *phitToTuple;
      auto const &foundNtuplets = *ptuples;
      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple->nbins())) {
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
          alpaka::atomicAdd(acc, &nshared[*it], 1ull, alpaka::hierarchy::Blocks{});
        }

      }  //  hit loop
    }
  };

  template <typename TrackerTraits>
  class Kernel_markSharedHit {
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  int const *__restrict__ nshared,
                                  HitContainer<TrackerTraits> const *__restrict__ tuples,
                                  Quality *__restrict__ quality,
                                  bool dupPassThrough) const {
      // constexpr auto bad = Quality::bad;
      constexpr auto dup = Quality::dup;
      constexpr auto loose = Quality::loose;
      // constexpr auto strict = Quality::strict;

      // quality to mark rejected
      auto const reject = dupPassThrough ? loose : dup;
      for (auto idx : cms::alpakatools::uniform_elements(acc, tuples->nbins())) {
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
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  uint16_t nmin,
                                  bool dupPassThrough,
                                  HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        auto score = [&](auto it, auto nl) { return std::abs(reco::tip(tracks_view, it)); };

        // full combinatorics
        for (auto ip = hitToTuple.begin(idx); ip < hitToTuple.end(idx) - 1; ++ip) {
          auto const it = *ip;
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;
          auto opi = tracks_view[it].state()(2);
          auto e2opi = tracks_view[it].covariance()(9);
          auto cti = tracks_view[it].state()(3);
          auto e2cti = tracks_view[it].covariance()(12);
          auto nli = tracks_view[it].nLayers();
          for (auto jp = ip + 1; jp < hitToTuple.end(idx); ++jp) {
            auto const jt = *jp;
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            auto opj = tracks_view[jt].state()(2);
            auto ctj = tracks_view[jt].state()(3);
            auto dct = nSigma2 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto dop = nSigma2 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            auto nlj = tracks_view[jt].nLayers();
            if (nlj < nli || (nlj == nli && (qj < qi || (qj == qi && score(it, nli) < score(jt, nlj)))))
              tracks_view[jt].quality() = reject;
            else {
              tracks_view[it].quality() = reject;
              break;
            }
          }
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_sharedHitCleaner {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  HitsConstView<TrackerTraits> hh,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  int nmin,
                                  bool dupPassThrough,
                                  HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      // quality of longest track
      auto const longTqual = Quality::highPurity;

      auto &hitToTuple = *phitToTuple;

      uint32_t l1end = hh.hitsLayerStart()[1];

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
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

          //checking if shared hit is on bpix1 and if the tuple is short enough
          if (idx < l1end and nl > nmin)
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
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  uint16_t nmin,
                                  bool dupPassThrough,
                                  HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::strict;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        float mc = maxScore;
        uint16_t im = tkNotFound;
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
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  uint16_t nmin,
                                  bool dupPassThrough,
                                  HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::loose;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        float mc = maxScore;
        uint16_t im = tkNotFound;

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
          if (tracks_view[it].quality() > reject && tracks_view.hitIndices().size(it) < 4 && it != im)
            tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
        }

      }  // loop over hits
    }
  };

  template <typename TrackerTraits>
  class Kernel_print_found_ntuplets {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  HitsConstView<TrackerTraits> hh,
                                  TkSoAView<TrackerTraits> tracks_view,
                                  HitToTuple<TrackerTraits> const *__restrict__ phitToTuple,
                                  int32_t firstPrint,
                                  int32_t lastPrint,
                                  int iev) const {
      constexpr auto loose = Quality::loose;

      for (auto i :
           cms::alpakatools::uniform_elements(acc, firstPrint, std::min(lastPrint, tracks_view.hitIndices().nbins()))) {
        auto nh = tracks_view.hitIndices().size(i);
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
               hh[*tracks_view.hitIndices().begin(i)].zGlobal(),
               hh[*(tracks_view.hitIndices().begin(i) + 1)].zGlobal(),
               hh[*(tracks_view.hitIndices().begin(i) + 2)].zGlobal(),
               nh > 3 ? hh[int(*(tracks_view.hitIndices().begin(i) + 3))].zGlobal() : 0,
               nh > 4 ? hh[int(*(tracks_view.hitIndices().begin(i) + 4))].zGlobal() : 0,
               nh > 5 ? hh[int(*(tracks_view.hitIndices().begin(i) + 5))].zGlobal() : 0,
               nh > 6 ? hh[int(*(tracks_view.hitIndices().begin(i) + nh - 1))].zGlobal() : 0);
      }
    }
  };

  class Kernel_printCounters {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc, Counters const *counters) const {
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
