//
// Original Author: Felice Pantaleo, CERN
//

// #define NTUPLE_DEBUG
// #define GPU_DEBUG

#include <cmath>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"

#include "CAStructures.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

namespace caHitNtupletGeneratorKernels {

  constexpr uint32_t tkNotFound = std::numeric_limits<uint16_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  constexpr float nSigma2 = 25.f;

  //all of these below are mostly to avoid brining around the relative namespace

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

  using Quality = pixelTrack::Quality;

  template <typename TrackerTraits>
  using TkSoAView = TrackSoAView<TrackerTraits>;

  template <typename TrackerTraits>
  using HitContainer = typename TrackSoA<TrackerTraits>::HitContainer;

  template <typename TrackerTraits>
  using HitsConstView = typename GPUCACellT<TrackerTraits>::HitsConstView;

  template <typename TrackerTraits>
  using QualityCuts = pixelTrack::QualityCutsT<TrackerTraits>;

  template <typename TrackerTraits>
  using CAParams = caHitNtupletGenerator::CAParamsT<TrackerTraits>;

  using Counters = caHitNtupletGenerator::Counters;

  template <typename TrackerTraits>
  __global__ void kernel_checkOverflows(TkSoAView<TrackerTraits> tracks_view,
                                        TupleMultiplicity<TrackerTraits> const *tupleMultiplicity,
                                        HitToTuple<TrackerTraits> const *hitToTuple,
                                        cms::cuda::AtomicPairCounter *apc,
                                        GPUCACellT<TrackerTraits> const *__restrict__ cells,
                                        uint32_t const *__restrict__ nCells,
                                        CellNeighborsVector<TrackerTraits> const *cellNeighbors,
                                        CellTracksVector<TrackerTraits> const *cellTracks,
                                        OuterHitOfCell<TrackerTraits> const isOuterHitOfCell,
                                        int32_t nHits,
                                        uint32_t maxNumberOfDoublets,
                                        Counters *counters) {
    auto first = threadIdx.x + blockIdx.x * blockDim.x;

    auto &c = *counters;
    // counters once per event
    if (0 == first) {
      atomicAdd(&c.nEvents, 1);
      atomicAdd(&c.nHits, nHits);
      atomicAdd(&c.nCells, *nCells);
      atomicAdd(&c.nTuples, apc->get().m);
      atomicAdd(&c.nFitTracks, tupleMultiplicity->size());
    }

#ifdef NTUPLE_DEBUG
    if (0 == first) {
      printf("number of found cells %d \n found tuples %d with total hits %d out of %d %d\n",
             *nCells,
             apc->get().m,
             apc->get().n,
             nHits,
             hitToTuple->totOnes());
      if (apc->get().m < TrackerTraits::maxNumberOfQuadruplets) {
        assert(tracks_view.hitIndices().size(apc->get().m) == 0);
        assert(tracks_view.hitIndices().size() == apc->get().n);
      }
    }

    for (int idx = first, nt = tracks_view.hitIndices().nOnes(); idx < nt; idx += gridDim.x * blockDim.x) {
      if (tracks_view.hitIndices().size(idx) > TrackerTraits::maxHitsOnTrack)  // current real limit
        printf("ERROR %d, %d\n", idx, tracks_view.hitIndices().size(idx));
      assert(tracks_view.hitIndices().size(idx) <= TrackerTraits::maxHitsOnTrack);
      for (auto ih = tracks_view.hitIndices().begin(idx); ih != tracks_view.hitIndices().end(idx); ++ih)
        assert(int(*ih) < nHits);
    }
#endif

    if (0 == first) {
      if (apc->get().m >= TrackerTraits::maxNumberOfQuadruplets)
        printf("Tuples overflow\n");
      if (*nCells >= maxNumberOfDoublets)
        printf("Cells overflow\n");
      if (cellNeighbors && cellNeighbors->full())
        printf("cellNeighbors overflow %d %d \n", cellNeighbors->capacity(), cellNeighbors->size());
      if (cellTracks && cellTracks->full())
        printf("cellTracks overflow\n");
      if (int(hitToTuple->nOnes()) < nHits)
        printf("ERROR hitToTuple  overflow %d %d\n", hitToTuple->nOnes(), nHits);
#ifdef GPU_DEBUG
      printf("size of cellNeighbors %d \n cellTracks %d \n hitToTuple %d \n",
             cellNeighbors->size(),
             cellTracks->size(),
             hitToTuple->size());

#endif
    }

    for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
      auto const &thisCell = cells[idx];
      if (thisCell.hasFishbone() && !thisCell.isKilled())
        atomicAdd(&c.nFishCells, 1);
      if (thisCell.outerNeighbors().full())  //++tooManyNeighbors[thisCell.theLayerPairId];
        printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.layerPairId());
      if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
        printf("Tracks overflow %d in %d\n", idx, thisCell.layerPairId());
      if (thisCell.isKilled())
        atomicAdd(&c.nKilledCells, 1);
      if (!thisCell.unused())
        atomicAdd(&c.nEmptyCells, 1);
      if ((0 == hitToTuple->size(thisCell.inner_hit_id())) && (0 == hitToTuple->size(thisCell.outer_hit_id())))
        atomicAdd(&c.nZeroTrackCells, 1);
    }

    for (int idx = first, nt = nHits - isOuterHitOfCell.offset; idx < nt; idx += gridDim.x * blockDim.x) {
      if (isOuterHitOfCell.container[idx].full())  // ++tooManyOuterHitOfCell;
        printf("OuterHitOfCell overflow %d\n", idx);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_fishboneCleaner(GPUCACellT<TrackerTraits> const *cells,
                                         uint32_t const *__restrict__ nCells,
                                         TkSoAView<TrackerTraits> tracks_view) {
    constexpr auto reject = pixelTrack::Quality::dup;

    auto first = threadIdx.x + blockIdx.x * blockDim.x;
    for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
      auto const &thisCell = cells[idx];
      if (!thisCell.isKilled())
        continue;

      for (auto it : thisCell.tracks())
        tracks_view[it].quality() = reject;
    }
  }

  // remove shorter tracks if sharing a cell
  // It does not seem to affect efficiency in any way!
  template <typename TrackerTraits>
  __global__ void kernel_earlyDuplicateRemover(GPUCACellT<TrackerTraits> const *cells,
                                               uint32_t const *__restrict__ nCells,
                                               TkSoAView<TrackerTraits> tracks_view,
                                               bool dupPassThrough) {
    // quality to mark rejected
    constexpr auto reject = pixelTrack::Quality::edup;  /// cannot be loose

    assert(nCells);
    auto first = threadIdx.x + blockIdx.x * blockDim.x;
    for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
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
      // quad pass through (leave it her for tests)
      //  maxNl = std::min(4, maxNl);

      for (auto it : thisCell.tracks()) {
        if (tracks_view[it].nLayers() < maxNl)
          tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
      }
    }
  }

  // assume the above (so, short tracks already removed)
  template <typename TrackerTraits>
  __global__ void kernel_fastDuplicateRemover(GPUCACellT<TrackerTraits> const *__restrict__ cells,
                                              uint32_t const *__restrict__ nCells,
                                              TkSoAView<TrackerTraits> tracks_view,
                                              bool dupPassThrough) {
    // quality to mark rejected
    auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;
    constexpr auto loose = pixelTrack::Quality::loose;

    assert(nCells);

    auto first = threadIdx.x + blockIdx.x * blockDim.x;
    for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
      auto const &thisCell = cells[idx];
      if (thisCell.tracks().size() < 2)
        continue;

      float mc = maxScore;
      uint16_t im = tkNotFound;

      auto score = [&](auto it) { return std::abs(TracksUtilities<TrackerTraits>::tip(tracks_view, it)); };

      // full crazy combinatorics
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

  template <typename TrackerTraits>
  __global__ void kernel_connect(cms::cuda::AtomicPairCounter *apc1,
                                 cms::cuda::AtomicPairCounter *apc2,  // just to zero them,
                                 HitsConstView<TrackerTraits> hh,
                                 GPUCACellT<TrackerTraits> *cells,
                                 uint32_t const *__restrict__ nCells,
                                 CellNeighborsVector<TrackerTraits> *cellNeighbors,
                                 OuterHitOfCell<TrackerTraits> const isOuterHitOfCell,
                                 CAParams<TrackerTraits> params) {
    using Cell = GPUCACellT<TrackerTraits>;

    auto firstCellIndex = threadIdx.y + blockIdx.y * blockDim.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;

    if (0 == (firstCellIndex + first)) {
      (*apc1) = 0;
      (*apc2) = 0;
    }  // ready for next kernel

    constexpr uint32_t last_bpix1_detIndex = TrackerTraits::last_bpix1_detIndex;
    constexpr uint32_t last_barrel_detIndex = TrackerTraits::last_barrel_detIndex;
    for (int idx = firstCellIndex, nt = (*nCells); idx < nt; idx += gridDim.y * blockDim.y) {
      auto cellIndex = idx;
      auto &thisCell = cells[idx];
      auto innerHitId = thisCell.inner_hit_id();
      if (int(innerHitId) < isOuterHitOfCell.offset)
        continue;
      int numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
      auto vi = isOuterHitOfCell[innerHitId].data();

      auto ri = thisCell.inner_r(hh);
      auto zi = thisCell.inner_z(hh);

      auto ro = thisCell.outer_r(hh);
      auto zo = thisCell.outer_z(hh);
      auto isBarrel = thisCell.inner_detIndex(hh) < last_barrel_detIndex;

      for (int j = first; j < numberOfPossibleNeighbors; j += stride) {
        auto otherCell = __ldg(vi + j);
        auto &oc = cells[otherCell];
        auto r1 = oc.inner_r(hh);
        auto z1 = oc.inner_z(hh);
        bool aligned = Cell::areAlignedRZ(
            r1,
            z1,
            ri,
            zi,
            ro,
            zo,
            params.ptmin_,
            isBarrel ? params.CAThetaCutBarrel_ : params.CAThetaCutForward_);  // 2.f*thetaCut); // FIXME tune cuts
        if (aligned && thisCell.dcaCut(hh,
                                       oc,
                                       oc.inner_detIndex(hh) < last_bpix1_detIndex ? params.dcaCutInnerTriplet_
                                                                                   : params.dcaCutOuterTriplet_,
                                       params.hardCurvCut_)) {  // FIXME tune cuts
          oc.addOuterNeighbor(cellIndex, *cellNeighbors);
          thisCell.setStatusBits(Cell::StatusBit::kUsed);
          oc.setStatusBits(Cell::StatusBit::kUsed);
        }
      }  // loop on inner cells
    }    // loop on outer cells
  }

  template <typename TrackerTraits>
  __global__ void kernel_find_ntuplets(HitsConstView<TrackerTraits> hh,
                                       TkSoAView<TrackerTraits> tracks_view,
                                       GPUCACellT<TrackerTraits> *__restrict__ cells,
                                       uint32_t const *nCells,
                                       CellTracksVector<TrackerTraits> *cellTracks,
                                       cms::cuda::AtomicPairCounter *apc,
                                       CAParams<TrackerTraits> params) {
    // recursive: not obvious to widen

    using Cell = GPUCACellT<TrackerTraits>;

    auto first = threadIdx.x + blockIdx.x * blockDim.x;

#ifdef GPU_DEBUG
    if (first == 0)
      printf("starting producing ntuplets from %d cells \n", *nCells);
#endif
    for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
      auto const &thisCell = cells[idx];

      if (thisCell.isKilled())
        continue;  // cut by earlyFishbone

      // we require at least three hits...
      if (thisCell.outerNeighbors().empty())
        continue;

      auto pid = thisCell.layerPairId();
      bool doit = params.startingLayerPair(pid);

      constexpr uint32_t maxDepth = TrackerTraits::maxDepth;
      if (doit) {
        typename Cell::TmpTuple stack;
        stack.reset();

        bool bpix1Start = params.startAt0(pid);

        thisCell.template find_ntuplets<maxDepth>(hh,
                                                  cells,
                                                  *cellTracks,
                                                  tracks_view.hitIndices(),
                                                  *apc,
                                                  tracks_view.quality(),
                                                  stack,
                                                  params.minHitsPerNtuplet_,
                                                  bpix1Start);

        assert(stack.empty());
      }
    }
  }
  template <typename TrackerTraits>
  __global__ void kernel_mark_used(GPUCACellT<TrackerTraits> *__restrict__ cells, uint32_t const *nCells) {
    auto first = threadIdx.x + blockIdx.x * blockDim.x;
    using Cell = GPUCACellT<TrackerTraits>;
    for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
      auto &thisCell = cells[idx];
      if (!thisCell.tracks().empty())
        thisCell.setStatusBits(Cell::StatusBit::kInTrack);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_countMultiplicity(TkSoAView<TrackerTraits> tracks_view,
                                           TupleMultiplicity<TrackerTraits> *tupleMultiplicity) {
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = first, nt = tracks_view.hitIndices().nOnes(); it < nt; it += gridDim.x * blockDim.x) {
      auto nhits = tracks_view.hitIndices().size(it);
      if (nhits < 3)
        continue;
      if (tracks_view[it].quality() == pixelTrack::Quality::edup)
        continue;
      assert(tracks_view[it].quality() == pixelTrack::Quality::bad);
      if (nhits > TrackerTraits::maxHitsOnTrack)  // current limit
        printf("wrong mult %d %d\n", it, nhits);
      assert(nhits <= TrackerTraits::maxHitsOnTrack);
      tupleMultiplicity->count(nhits);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_fillMultiplicity(TkSoAView<TrackerTraits> tracks_view,
                                          TupleMultiplicity<TrackerTraits> *tupleMultiplicity) {
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = first, nt = tracks_view.hitIndices().nOnes(); it < nt; it += gridDim.x * blockDim.x) {
      auto nhits = tracks_view.hitIndices().size(it);
      if (nhits < 3)
        continue;
      if (tracks_view[it].quality() == pixelTrack::Quality::edup)
        continue;
      assert(tracks_view[it].quality() == pixelTrack::Quality::bad);
      if (nhits > TrackerTraits::maxHitsOnTrack)
        printf("wrong mult %d %d\n", it, nhits);
      assert(nhits <= TrackerTraits::maxHitsOnTrack);
      tupleMultiplicity->fill(nhits, it);
    }
  }

  ///TODO : why there was quality here?
  template <typename TrackerTraits>
  __global__ void kernel_classifyTracks(TkSoAView<TrackerTraits> tracks_view, QualityCuts<TrackerTraits> cuts) {
    // Quality *__restrict__ quality) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int it = first, nt = tracks_view.hitIndices().nOnes(); it < nt; it += gridDim.x * blockDim.x) {
      auto nhits = tracks_view.hitIndices().size(it);
      if (nhits == 0)
        break;  // guard

      // if duplicate: not even fit
      if (tracks_view[it].quality() == pixelTrack::Quality::edup)
        continue;

      assert(tracks_view[it].quality() == pixelTrack::Quality::bad);

      // mark doublets as bad
      if (nhits < 3)
        continue;

      // if the fit has any invalid parameters, mark it as bad
      bool isNaN = false;
      for (int i = 0; i < 5; ++i) {
        isNaN |= std::isnan(tracks_view[it].state()(i));
      }
      if (isNaN) {
#ifdef NTUPLE_DEBUG
        printf("NaN in fit %d size %d chi2 %f\n", it, tracks_view.hitIndices().size(it), tracks_view[it].chi2());
#endif
        continue;
      }

      tracks_view[it].quality() = pixelTrack::Quality::strict;

      if (cuts.strictCut(tracks_view, it))
        continue;

      tracks_view[it].quality() = pixelTrack::Quality::tight;

      if (cuts.isHP(tracks_view, nhits, it))
        tracks_view[it].quality() = pixelTrack::Quality::highPurity;
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_doStatsForTracks(TkSoAView<TrackerTraits> tracks_view, Counters *counters) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = tracks_view.hitIndices().nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (tracks_view.hitIndices().size(idx) == 0)
        break;  //guard
      if (tracks_view[idx].quality() < pixelTrack::Quality::loose)
        continue;
      atomicAdd(&(counters->nLooseTracks), 1);
      if (tracks_view[idx].quality() < pixelTrack::Quality::strict)
        continue;
      atomicAdd(&(counters->nGoodTracks), 1);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_countHitInTracks(TkSoAView<TrackerTraits> tracks_view, HitToTuple<TrackerTraits> *hitToTuple) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = tracks_view.hitIndices().nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (tracks_view.hitIndices().size(idx) == 0)
        break;  // guard
      for (auto h = tracks_view.hitIndices().begin(idx); h != tracks_view.hitIndices().end(idx); ++h)
        hitToTuple->count(*h);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_fillHitInTracks(TkSoAView<TrackerTraits> tracks_view, HitToTuple<TrackerTraits> *hitToTuple) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = tracks_view.hitIndices().nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (tracks_view.hitIndices().size(idx) == 0)
        break;  // guard
      for (auto h = tracks_view.hitIndices().begin(idx); h != tracks_view.hitIndices().end(idx); ++h)
        hitToTuple->fill(*h, idx);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_fillHitDetIndices(TkSoAView<TrackerTraits> tracks_view, HitsConstView<TrackerTraits> hh) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    // copy offsets
    for (int idx = first, ntot = tracks_view.hitIndices().totOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      tracks_view.detIndices().off[idx] = tracks_view.hitIndices().off[idx];
    }
    // fill hit indices
    auto nhits = hh.nHits();

    for (int idx = first, ntot = tracks_view.hitIndices().size(); idx < ntot; idx += gridDim.x * blockDim.x) {
      assert(tracks_view.hitIndices().content[idx] < nhits);
      tracks_view.detIndices().content[idx] = hh[tracks_view.hitIndices().content[idx]].detectorIndex();
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_fillNLayers(TkSoAView<TrackerTraits> tracks_view, cms::cuda::AtomicPairCounter *apc) {
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    // clamp the number of tracks to the capacity of the SoA
    auto ntracks = std::min<int>(apc->get().m, tracks_view.metadata().size() - 1);
    if (0 == first)
      tracks_view.nTracks() = ntracks;
    for (int idx = first, nt = ntracks; idx < nt; idx += gridDim.x * blockDim.x) {
      auto nHits = TracksUtilities<TrackerTraits>::nHits(tracks_view, idx);
      assert(nHits >= 3);
      tracks_view[idx].nLayers() = TracksUtilities<TrackerTraits>::computeNumberOfLayers(tracks_view, idx);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_doStatsForHitInTracks(HitToTuple<TrackerTraits> const *__restrict__ hitToTuple,
                                               Counters *counters) {
    auto &c = *counters;
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = hitToTuple->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (hitToTuple->size(idx) == 0)
        continue;  // SHALL NOT BE break
      atomicAdd(&c.nUsedHits, 1);
      if (hitToTuple->size(idx) > 1)
        atomicAdd(&c.nDupHits, 1);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_countSharedHit(int *__restrict__ nshared,
                                        HitContainer<TrackerTraits> const *__restrict__ ptuples,
                                        Quality const *__restrict__ quality,
                                        HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
    constexpr auto loose = pixelTrack::Quality::loose;

    auto &hitToTuple = *phitToTuple;
    auto const &foundNtuplets = *ptuples;

    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
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
        atomicAdd(&nshared[*it], 1);
      }

    }  //  hit loop
  }

  template <typename TrackerTraits>
  __global__ void kernel_markSharedHit(int const *__restrict__ nshared,
                                       HitContainer<TrackerTraits> const *__restrict__ tuples,
                                       Quality *__restrict__ quality,
                                       bool dupPassThrough) {
    // constexpr auto bad = pixelTrack::Quality::bad;
    constexpr auto dup = pixelTrack::Quality::dup;
    constexpr auto loose = pixelTrack::Quality::loose;
    // constexpr auto strict = pixelTrack::Quality::strict;

    // quality to mark rejected
    auto const reject = dupPassThrough ? loose : dup;

    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = tuples->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (tuples->size(idx) == 0)
        break;  //guard
      if (quality[idx] <= reject)
        continue;
      if (nshared[idx] > 2)
        quality[idx] = reject;
    }
  }

  // mostly for very forward triplets.....
  template <typename TrackerTraits>
  __global__ void kernel_rejectDuplicate(TkSoAView<TrackerTraits> tracks_view,
                                         uint16_t nmin,
                                         bool dupPassThrough,
                                         HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
    // quality to mark rejected
    auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;

    auto &hitToTuple = *phitToTuple;

    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (hitToTuple.size(idx) < 2)
        continue;

      auto score = [&](auto it, auto nl) { return std::abs(TracksUtilities<TrackerTraits>::tip(tracks_view, it)); };

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

  template <typename TrackerTraits>
  __global__ void kernel_sharedHitCleaner(HitsConstView<TrackerTraits> hh,
                                          TkSoAView<TrackerTraits> tracks_view,
                                          int nmin,
                                          bool dupPassThrough,
                                          HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
    // quality to mark rejected
    auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;
    // quality of longest track
    auto const longTqual = pixelTrack::Quality::highPurity;

    auto &hitToTuple = *phitToTuple;

    int l1end = hh.hitsLayerStart()[1];

    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
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

  template <typename TrackerTraits>
  __global__ void kernel_tripletCleaner(TkSoAView<TrackerTraits> tracks_view,
                                        uint16_t nmin,
                                        bool dupPassThrough,
                                        HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
    // quality to mark rejected
    auto const reject = pixelTrack::Quality::loose;
    /// min quality of good
    auto const good = pixelTrack::Quality::strict;

    auto &hitToTuple = *phitToTuple;

    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (hitToTuple.size(idx) < 2)
        continue;

      float mc = maxScore;
      uint16_t im = tkNotFound;
      bool onlyTriplets = true;

      // check if only triplets
      for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
        if (tracks_view[*it].quality() <= good)
          continue;
        onlyTriplets &= TracksUtilities<TrackerTraits>::isTriplet(tracks_view, *it);
        if (!onlyTriplets)
          break;
      }

      // only triplets
      if (!onlyTriplets)
        continue;

      // for triplets choose best tip!  (should we first find best quality???)
      for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
        auto const it = *ip;
        if (tracks_view[it].quality() >= good && std::abs(TracksUtilities<TrackerTraits>::tip(tracks_view, it)) < mc) {
          mc = std::abs(TracksUtilities<TrackerTraits>::tip(tracks_view, it));
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

  template <typename TrackerTraits>
  __global__ void kernel_simpleTripletCleaner(TkSoAView<TrackerTraits> tracks_view,
                                              uint16_t nmin,
                                              bool dupPassThrough,
                                              HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
    // quality to mark rejected
    auto const reject = pixelTrack::Quality::loose;
    /// min quality of good
    auto const good = pixelTrack::Quality::loose;

    auto &hitToTuple = *phitToTuple;

    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (hitToTuple.size(idx) < 2)
        continue;

      float mc = maxScore;
      uint16_t im = tkNotFound;

      // choose best tip!  (should we first find best quality???)
      for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
        auto const it = *ip;
        if (tracks_view[it].quality() >= good && std::abs(TracksUtilities<TrackerTraits>::tip(tracks_view, it)) < mc) {
          mc = std::abs(TracksUtilities<TrackerTraits>::tip(tracks_view, it));
          im = it;
        }
      }

      if (tkNotFound == im)
        continue;

      // mark worse ambiguities
      for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
        auto const it = *ip;
        if (tracks_view[it].quality() > reject && TracksUtilities<TrackerTraits>::isTriplet(tracks_view, it) &&
            it != im)
          tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
      }

    }  // loop over hits
  }

  template <typename TrackerTraits>
  __global__ void kernel_print_found_ntuplets(HitsConstView<TrackerTraits> hh,
                                              TkSoAView<TrackerTraits> tracks_view,
                                              HitToTuple<TrackerTraits> const *__restrict__ phitToTuple,
                                              int32_t firstPrint,
                                              int32_t lastPrint,
                                              int iev) {
    constexpr auto loose = pixelTrack::Quality::loose;

    int first = firstPrint + blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = first, np = std::min(lastPrint, tracks_view.hitIndices().nOnes()); i < np;
         i += blockDim.x * gridDim.x) {
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
             TracksUtilities<TrackerTraits>::charge(tracks_view, i),
             tracks_view[i].pt(),
             tracks_view[i].eta(),
             TracksUtilities<TrackerTraits>::phi(tracks_view, i),
             TracksUtilities<TrackerTraits>::tip(tracks_view, i),
             TracksUtilities<TrackerTraits>::zip(tracks_view, i),
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

  __global__ void kernel_printCounters(Counters const *counters) {
    auto const &c = *counters;
    printf(
        "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nLooseTracks  |  nGoodTracks | nUsedHits | "
        "nDupHits | "
        "nFishCells | "
        "nKilledCells | "
        "nUsedCells | nZeroTrackCells ||\n");
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
        "Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f|  %.3f|  %.3f||\n",
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

}  // namespace caHitNtupletGeneratorKernels
