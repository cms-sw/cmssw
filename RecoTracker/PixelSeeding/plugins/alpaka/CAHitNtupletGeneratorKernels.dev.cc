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
#include "CAHitNtupletGeneratorKernelsImpl.h"

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

    // CELL BOUND. A separate name from maxDoublets because the two are not the same thing: this one
    // bounds the CELL ARRAY and the hit->cell association that carries one entry per cell, while
    // maxDoublets is ALSO the basis of the content capacities (nCellsToCells, nCellsToTracks) and of the
    // key-space extent nBins, all of which are computed in allocateAfterDoublets and keep following the
    // cap formula whatever this bound does.
    //
    // The bound is capped at an allocator bin edge. maxDoublets is an envelope fit of the doublet demand;
    // at its largest values the cell array (sizeof(SimpleCell) = 12 B) would be served from the caching
    // allocator's next power-of-two bin although it fits the smaller one in almost every event, and the
    // allocator never gives a block back, so one event above the edge would pin the larger block for the
    // rest of the job. Capping the cell bound at the edge removes that bin; it is a capacity cut of at
    // most a few percent in the rare events whose formula value exceeds the edge, far above the observed
    // doublet demand. The content capacities are untouched: allocateAfterDoublets is still called with
    // maxDoublets, so nCellsToCells and nCellsToTracks -- the ratios the truncation guards compare
    // against -- do not move.
    static constexpr uint32_t kMaxDoubletsForCellBin = (64u * 1024u * 1024u) / uint32_t(sizeof(SimpleCell));
    const uint32_t cellBound = std::min(maxDoublets, kMaxDoubletsForCellBin);
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
                        this->deviceTriplets_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_hitToCell_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->pipelineCountersPtr());

    CellToCell::template launchFinalize<Acc1D>(this->device_cellToNeighborsView_, queue);

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
                        this->device_hitTuple_apc_);

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
                        this->m_params.qualityCuts_);
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
                        Kernel_printSizes{},
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

  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase1>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2OT>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2OTStubs>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
