#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h

// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES
// Per-stage rejection accounting for the CA funnel (which cut killed each doublet/triplet/tuple,
// see CAPipelineCounters.h). Off by default and costs nothing when off.
// #define CA_PIPELINE_COUNTERS
// The per-event per-container sizing dump toggle (CA_SIZING_DUMP) lives in CASizingDumpMacro.h
// (included below), because other stages of the same dump have to see the same switch.

#include <cstdint>
#include <string>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAPairSoACollection.h"
// CA_SIZING_DUMP toggle (minimal header) -- the per-event demand dump of every sized container.
#include "CASizingDumpMacro.h"

#include "CACell.h"
#include "CAPipelineCounters.h"
#include "CAPixelDoublets.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::caStructures;

  namespace caHitNtupletGenerator {

    // Re-export pipeline counter enum from global namespace
    using namespace ::caHitNtupletGenerator;

    //Counters
    struct Counters {
      unsigned long long nEvents;
      unsigned long long nHits;
      unsigned long long nCells;
      unsigned long long nTuples;
      unsigned long long nFitTracks;
      unsigned long long nLooseTracks;
      unsigned long long nGoodTracks;
      unsigned long long nUsedHits;
      unsigned long long nDupHits;
      unsigned long long nFishCells;
      unsigned long long nKilledCells;
      unsigned long long nEmptyCells;
      unsigned long long nZeroTrackCells;
      // Per-container overflow counters (lossy truncation, counted not asserted).
      // The first four are per-event flags raised by Kernel_checkOverflows (doStats_ only): they
      // count events in which the container ended at or past its capacity. The last two are
      // per-entry drop counts written at the guard site itself, once per dropped association
      // (at the fill pass; the matching count pass skips the same keys).
      unsigned long long nTupleOverflow;       // hitContainer: tuple slots at/past capacity
      unsigned long long nCellOverflow;        // simpleCells/doublet store: nCells at maxDoublets
      unsigned long long nTripletOverflow;     // cellToCell (cellNeighbors): nTrips at maxTriplets
      unsigned long long nCellTrackOverflow;   // cellToTrack (cellTracks): nCellTracks at capacity
      unsigned long long nHitToTupleOverflow;  // hitToTuple: entries dropped, key past nOnes
      unsigned long long nHitToCellOverflow;   // hitToCell: entries dropped, key past nOnes
    };

    // Number of per-track hit rows to reserve for maxTuples tuples at the configured average
    // hits-per-track. This is the one expression behind both the internal hitContainer/hitToTuple
    // storage and the output trackHits SoA, so the two capacities are equal by construction: the
    // CA fills the container and Kernel_fillHitDetIndices copies it into the SoA clamped to the
    // SoA's row count, and a smaller SoA would truncate a container that did not overflow.
    // Rounded up, so a fractional avgHitsPerTrack is delivered in full rather than floored. The
    // product is formed in double: maxTuples reaches ~1e6 at PU200 and a float product would
    // round away more than the fraction being kept.
    inline uint32_t nHitRowsForTuples(uint32_t maxTuples, float avgHitsPerTrack) {
      const double rows = double(maxTuples) * double(avgHitsPerTrack);
      uint32_t n = (rows > 0.) ? uint32_t(rows) : 0u;
      if (double(n) < rows)
        ++n;
      return (n > 0u) ? n : 1u;
    }

    //Full list of params = algo params + quality cuts
    //Generic template
    template <typename TrackerTraits, typename Enable = void>
    struct ParamsT {};

    template <typename TrackerTraits>
    struct ParamsT<TrackerTraits, pixelTopology::isPhase1Topology<TrackerTraits>> {
      using TT = TrackerTraits;
      using QualityCuts = ::pixelTrack::QualityCutsT<TT>;  //track quality cuts

      ParamsT(AlgoParams const& commonCuts, QualityCuts const& qualityCuts)
          : algoParams_(commonCuts), qualityCuts_(qualityCuts) {}

      const AlgoParams algoParams_;
      const QualityCuts qualityCuts_{// polynomial coefficients for the pT-dependent chi2 cut
                                     {0.68177776, 0.74609577, -0.08035491, 0.00315399},
                                     // max pT used to determine the chi2 cut
                                     10.,
                                     // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                                     30.,
                                     // regional cuts for triplets
                                     {
                                         0.3,  // |Tip| < 0.3 cm
                                         0.5,  // pT > 0.5 GeV
                                         12.0  // |Zip| < 12.0 cm
                                     },
                                     // regional cuts for quadruplets
                                     {
                                         0.5,  // |Tip| < 0.5 cm
                                         0.3,  // pT > 0.3 GeV
                                         12.0  // |Zip| < 12.0 cm
                                     }};

    };  // Params Phase1

    template <typename TrackerTraits>
    struct ParamsT<TrackerTraits, pixelTopology::isPhase2Topology<TrackerTraits>> : public AlgoParams {
      using TT = TrackerTraits;
      using QualityCuts = ::pixelTrack::QualityCutsT<TT>;

      ParamsT(AlgoParams const& commonCuts, QualityCuts const& qualityCuts)
          : algoParams_(commonCuts), qualityCuts_(qualityCuts) {}

      // quality cuts
      const AlgoParams algoParams_;
      const QualityCuts qualityCuts_{5.0f, /*chi2*/ 0.9f, /* pT in Gev*/ 0.4f, /*zip in cm*/ 12.0f /*tip in cm*/};

    };  // Params Phase1

  }  // namespace caHitNtupletGenerator
  template <typename TTTraits>
  class CAHitNtupletGeneratorKernels {
  public:
    using TrackerTraits = TTTraits;

    using SimpleCell = CACell<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;
    // Track qualities
    using Quality = ::pixelTrack::Quality;
    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;

    // Histograms

    using PhiBinner = caStructures::PhiBinnerT<TrackerTraits>;  //the traits here define the number of layer/histograms
    using PhiBinnerStorageType = typename PhiBinner::value_type;
    using PhiBinnerView = typename PhiBinner::View;

    using HitToTuple = caStructures::GenericContainer;
    using HitContainer = caStructures::SequentialContainer;
    using TupleMultiplicity = caStructures::GenericContainer;
    using HitToCell = caStructures::GenericContainer;
    using CellToCell = caStructures::GenericContainer;
    using CellToTrack = caStructures::GenericContainer;

    using GenericContainer = caStructures::GenericContainer;
    using GenericContainerStorage = typename GenericContainer::value_type;
    using GenericContainerView = typename GenericContainer::View;
    using DeviceGenericContainerBuffer = std::optional<cms::alpakatools::device_buffer<Device, GenericContainer>>;
    using DeviceGenericStorageBuffer =
        std::optional<cms::alpakatools::device_buffer<Device, GenericContainerStorage[]>>;
    using DeviceGenericOffsetsBuffer =
        std::optional<cms::alpakatools::device_buffer<Device, GenericContainerOffsets[]>>;

    using SequentialContainer = caStructures::SequentialContainer;
    using SequentialContainerStorage = typename SequentialContainer::value_type;
    using SequentialContainerView = typename SequentialContainer::View;
    using DeviceSequentialContainerBuffer = std::optional<cms::alpakatools::device_buffer<Device, SequentialContainer>>;
    using DeviceSequentialStorageBuffer =
        std::optional<cms::alpakatools::device_buffer<Device, SequentialContainerStorage[]>>;
    using DeviceSequentialOffsetsBuffer =
        std::optional<cms::alpakatools::device_buffer<Device, SequentialContainerOffsets[]>>;

    CAHitNtupletGeneratorKernels(Params const& params,
                                 uint32_t nHits,
                                 uint32_t offsetBPIX2,
                                 uint32_t nDoublets,
                                 uint32_t nTracks,
                                 uint16_t nLayers,
                                 Queue& queue);
    ~CAHitNtupletGeneratorKernels() = default;

    TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_->data(); }
    // Finalized (prefix-scanned) per-N-bin cumulative offsets of the tuple-multiplicity assoc:
    // off[b] = number of fitted tuples with selected-hit count < b, so bin [nHitsL,nHitsH] holds
    // off[nHitsH+1]-off[nHitsL] tuples (== the totTK Kernel_BLFastFit breaks on). maxHitsOnTrack+2
    // valid entries. Surfaced for the main-fit chunk+bin-aware launch elision:
    // one D2H of this small array tells the host each bin's population without a census kernel.
    GenericContainerOffsets const* tupleMultiplicityOffsets() const { return device_tupleMultiplicityOffsets_->data(); }
    HitContainer const* hitContainer() const { return device_hitContainer_->data(); }
    PhiBinner const* hitPhiHist() const { return device_hitPhiHist_->data(); }
    HitToCell const* hitToCell() const { return device_hitToCell_->data(); }

    // Pipeline counter pointer: returns device pointer when enabled, nullptr otherwise
    uint32_t* pipelineCountersPtr() {
#ifdef CA_PIPELINE_COUNTERS
      return device_pipelineCounters_->data();
#else
      return nullptr;
#endif
    }
    HitToTuple const* hitToTuple() const { return device_hitToTuple_->data(); }
    CellToCell const* cellToCell() const { return device_cellToNeighbors_->data(); }
    CellToTrack const* cellToTrack() const { return device_cellToTracks_->data(); }

    void prepareHits(const HitsMultiView& hh,
                     const ModulesMultiView& mm,
                     const ::reco::CALayersSoAConstView& ll,
                     Queue& queue);

    // Persistent overflow accumulator (8 uint32; see Kernel_overflowSentinel's layout comment),
    // owned by CAHitNtupletGenerator per stream; when set, classifyTuples launches the always-on
    // Kernel_overflowSentinel into it (doStats-independent overflow surfacing).
    uint32_t* ovfAccum_ = nullptr;

    void launchKernels(const HitsMultiView& hh,
                       uint32_t offsetBPIX2,
                       uint16_t nLayers,
                       TkSoABlocksView& view,
                       const ::reco::CALayersSoAConstView& ll,
                       const ::reco::CAGraphSoAConstView& cc,
                       const ::reco::CATripletCutsSoAConstView& tripletCuts,
                       const ::reco::CANtupletCutsSoAConstView& ntupletCuts,
                       Queue& queue);

    void classifyTuples(const HitsMultiView& hh, TkSoAView& track_view, Queue& queue);

    // Returns the doublet count the countDoubletsFirst count-only pass read back to the host
    // (0 when countDoubletsFirst_ is off, or when the event was too empty to run the passes).
    // The caller reuses it as the allocateAfterDoublets sizing basis instead of paying a second
    // readbackNCells for the same scalar: the count pass skips the hit->cell capacity term
    // (CAPixelDoubletsAlgos.h, the `!CountOnly and ind >= capacity()` clause), so the fill pass
    // can never exceed it. The two values are equal whenever nothing overflows.
    uint32_t buildDoublets(const HitsMultiView& hh,
                           const ::reco::CAGraphSoAConstView& cc,
                           const ::reco::CALayersSoAConstView& ll,
                           const ::reco::CADoubletCutsSoAConstView& doubletCuts,
                           uint32_t offsetBPIX2,
                           Queue& queue);

    // Allocate the cell-derived buffers (cell->cell, cell->track, triplet/track-cell
    // SoA) once the actual number of doublets is known, sizing them from nCells
    // instead of the (much larger) maxDoublets safety cap. Called after buildDoublets.
    void allocateAfterDoublets(uint32_t nCells, Queue& queue);

    // Element count of the cells+offsets arena, in SimpleCell units, or 0 when the arena is not
    // worth it for this event's cap. Static so the sizing rule sits next to its two users.
    static uint32_t cellArenaExtent(uint32_t cellBound, uint32_t maxDoublets);

    // Allocate the hit->track storage from the actual number of hits-in-tracks,
    // known after launchKernels' finalizeBulk. Called before classifyTuples.
    void allocateAfterNtuplets(uint32_t nHitsInTracks, Queue& queue);

    // Return the BUILD-ONLY scratch to the caching allocator at the end of launchKernels, i.e. about
    // half an event before this object is destroyed. Stream-ordered and therefore sync-free: the
    // allocator records an event on the queue and only re-issues the block once that event has
    // completed, so the kernels already enqueued keep valid pointers. See the definition for the
    // per-buffer "last reader" argument. A no-op in the diagnostic builds (GPU_DEBUG / CA_STATS),
    // whose reports read these buffers' extents after launchKernels has returned.
    void releaseBuildScratch();

    // Read back the number of doublets (device_nCells_) to the host. One sync.
    uint32_t readbackNCells(Queue& queue);

    // Enqueue an async D2H of the 5-word extraStorage counter block into a caller-owned pinned
    // buffer. No wait is issued: the caller must synchronize its queue before reading (the CA
    // producer consumes it on the far side of the acquire->produce seam). Word [0] is the tuple
    // AtomicPairCounter (low half = tuple count, high half = hits-in-tracks total).
    void enqueueCountsReadback(cms::alpakatools::host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>& dst,
                               Queue& queue);

    // Read the whole 5-word extraStorage back to the host in one sync. Word [0] is the
    // AtomicPairCounter (first = tuple count, second = hits-in-tracks total), word [1] is unused,
    // words [2..4] are nCells, nTriplets and nCellTracks. Used by the CA_SIZING_DUMP demand dump.
    void readbackAllCounts(Queue& queue,
                           uint32_t& nTracks,
                           uint32_t& nHitsInTracks,
                           uint32_t& nCells,
                           uint32_t& nTriplets,
                           uint32_t& nCellTracks);

    // SoA element counts chosen in allocateAfterDoublets (cell->cell and cell->track edge list
    // capacities). Read by the CA_SIZING_DUMP per-event demand dump.
    uint32_t tripletsN() const { return tripletsN_; }
    uint32_t tracksCellsN() const { return tracksCellsN_; }

    static void printCounters();

  private:
    // params
    Params const& m_params;
    std::optional<cms::alpakatools::device_buffer<Device, Counters>> counters_;

    // Hits->Track
    DeviceGenericContainerBuffer device_hitToTuple_;
    DeviceGenericStorageBuffer device_hitToTupleStorage_;
    DeviceGenericOffsetsBuffer device_hitToTupleOffsets_;
    GenericContainerView device_hitToTupleView_;

    // (Outer) Hits-> Cells
    DeviceGenericContainerBuffer device_hitToCell_;
    DeviceGenericStorageBuffer device_hitToCellStorage_;
    DeviceGenericOffsetsBuffer device_hitToCellOffsets_;
    GenericContainerView device_hitToCellView_;

    // Hits Phi Binner
    std::optional<cms::alpakatools::device_buffer<Device, PhiBinner>> device_hitPhiHist_;
    std::optional<cms::alpakatools::device_buffer<Device, PhiBinnerStorageType[]>> device_phiBinnerStorage_;
    PhiBinnerView device_hitPhiView_;
    std::optional<cms::alpakatools::device_buffer<Device, hindex_type[]>> device_layerStarts_;

    // Scratch int32 mirror of the track quality used by the duplicate-removal kernels to make them
    // order/backend independent. The cell-parallel fast remover accumulates demotions here via atomicMin
    // and copies them back; the track-parallel hit-based removers instead use it as a frozen read-only
    // quality snapshot while each thread writes its own track's quality directly. See the helper kernels
    // Kernel_snapshotQuality / Kernel_applyQuality in CAHitNtupletGeneratorKernelsImpl.h
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> device_qualityScratch_;

    // Cells-> Neighbor Cells
    DeviceGenericContainerBuffer device_cellToNeighbors_;
    DeviceGenericStorageBuffer device_cellToNeighborsStorage_;
    DeviceGenericOffsetsBuffer device_cellToNeighborsOffsets_;
    GenericContainerView device_cellToNeighborsView_;

    // Cells-> Tracks
    DeviceGenericContainerBuffer device_cellToTracks_;
    DeviceGenericStorageBuffer device_cellToTracksStorage_;
    DeviceGenericOffsetsBuffer device_cellToTracksOffsets_;
    GenericContainerView device_cellToTracksView_;

    // Tracks->Hits
    DeviceSequentialContainerBuffer device_hitContainer_;
    DeviceGenericStorageBuffer device_hitContainerStorage_;
    DeviceSequentialOffsetsBuffer device_hitContainerOffsets_;
    SequentialContainerView device_hitContainerView_;
    // {offset bound, first dropped tuple} of the content-overflow repair.
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> device_tupleClampBound_;

    // No.Hits -> Track (Multiplicity)
    DeviceGenericContainerBuffer device_tupleMultiplicity_;
    DeviceGenericStorageBuffer device_tupleMultiplicityStorage_;
    DeviceGenericOffsetsBuffer device_tupleMultiplicityOffsets_;
    GenericContainerView device_tupleMultiplicityView_;

    std::optional<cms::alpakatools::device_buffer<Device, SimpleCell[]>> device_simpleCells_;

    // One allocation for two same-lifetime cell-scale buffers. device_simpleCells_ (12 B per cell)
    // and device_cellToTracksOffsets_ (4 B per cell + 1) are both sized from the doublet cap and
    // both live for the whole event: neither is released by releaseBuildScratch(), because the
    // duplicate removers read the cell->track container in classifyTuples. The alpaka
    // CachingAllocator rounds every request up to a power of two, so when 12*D + 4*D fits the bin
    // of 12*D alone, one shared buffer saves a bin; otherwise the two are allocated separately, so
    // the arena is never the more expensive option.
    // When the arena is in use, device_simpleCells_ IS the arena: its first maxDoublets entries are
    // the cells (every cell reader uses device_simpleCells_->data() unchanged) and the tail,
    // 256 B aligned, carries the cell->track offsets, whose pointer is handed to
    // allocateAfterDoublets through the member below. Null => allocate the offsets normally.
    caStructures::GenericContainerOffsets* arenaCellToTracksOffsets_ = nullptr;

    // Second layout (see chooseCellLayout). When the cells + track-offsets pairing above is not the
    // cheaper packing, the three cell-indexed uint32 arrays -- the hit->cell storage and the two
    // cell-keyed offsets -- share a buffer instead: 3 x 4*D packs into one bin whenever 12*D still
    // fits it, the large-event regime where each of them separately costs a full block. In this
    // layout releaseBuildScratch() keeps the first two alive together with the arena.
    std::optional<cms::alpakatools::device_buffer<Device, caStructures::GenericContainerStorage[]>>
        device_cellIndexArena_;
    caStructures::GenericContainerStorage* arenaHitToCellStorage_ = nullptr;
    caStructures::GenericContainerOffsets* arenaCellToNeighborsOffsets_ = nullptr;

    // Which of the three packings the constructor chose for this event.
    enum class CellLayout { kSeparate, kCellsWithTrackOffsets, kThreeCellIndexArrays };
    static CellLayout chooseCellLayout(uint32_t cellBound, uint32_t maxDoublets);

    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::AtomicPairCounter::DoubleWord[]>>
        device_extraStorage_;
    cms::alpakatools::AtomicPairCounter* device_hitTuple_apc_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> device_nCells_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> device_nTriplets_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> device_nCellTracks_;

    std::optional<CAPairSoACollection> deviceTriplets_;
    std::optional<CAPairSoACollection> deviceTracksCells_;

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters for diagnostic funnel
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> device_pipelineCounters_;
#endif

    // this could be inferred from the above buffers
    // but seems cleaner to have a dedicate variable
    uint32_t maxNumberOfDoublets_;

    // Work-division bound for the cell-loop kernels: the exact cell count when
    // countDoubletsFirst_ read it back, the doublet capacity otherwise. Their loops are
    // bounded by the true counts and any extent >= the count visits indices in the same
    // order, so the two settings produce identical output; the exact bound just launches
    // fewer idle blocks. Capacity checks keep using maxNumberOfDoublets_.
    uint32_t launchCells_ = 0;

    // SoA element counts chosen in allocateAfterDoublets, kept so the GPU_DEBUG
    // allocation report can size the SoA collections without recomputation.
    uint32_t tripletsN_ = 0;
    uint32_t tracksCellsN_ = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
