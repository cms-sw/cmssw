#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h

//#define GPU_DEBUG
//#define DUMP_GPU_TK_TUPLES

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/CAParamsSoA.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CACoupleSoACollection.h"

#include "CACell.h"
#include "CAPixelDoublets.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::caStructures;

  namespace caHitNtupletGenerator {

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
  };

    //Full list of params = algo params + ca params + cell params + quality cuts
  //Generic template
  template <typename TrackerTraits, typename Enable = void>
  struct ParamsT {
  };

  template <typename TrackerTraits>
  struct ParamsT<TrackerTraits, pixelTopology::isPhase1Topology<TrackerTraits>> {
    using TT = TrackerTraits;
    using QualityCuts = ::pixelTrack::QualityCutsT<TT>;  //track quality cuts

    ParamsT(AlgoParams const& commonCuts,
            QualityCuts const& qualityCuts)
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

    ParamsT(AlgoParams const& commonCuts,
            QualityCuts const& qualityCuts)
        : algoParams_(commonCuts), qualityCuts_(qualityCuts) {}

    // quality cuts
    const AlgoParams algoParams_;
    const QualityCuts qualityCuts_{5.0f, /*chi2*/ 0.9f, /* pT in Gev*/ 0.4f, /*zip in cm*/ 12.0f /*tip in cm*/};

  };  // Params Phase1

  }
  template <typename TTTraits>
  class CAHitNtupletGeneratorKernels {
    
  public:
    using TrackerTraits = TTTraits;
  
    // Cells containers
    using CellNeighborsVector = CellNeighborsVectorT<TrackerTraits>;
    using CellNeighbors = CellNeighborsT<TrackerTraits>;
    using CellTracksVector = CellTracksVectorT<TrackerTraits>;
    using CellTracks = CellTracksT<TrackerTraits>;
    using OuterHitOfCellContainer = OuterHitOfCellContainerT<TrackerTraits>;
    using OuterHitOfCell = OuterHitOfCellT<TrackerTraits>;

    using CACell = CACellT<TrackerTraits>;
    using SimpleCell = CASimpleCell<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;
    // Track qualities
    using Quality = ::pixelTrack::Quality;
    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;

    // Histograms
    /// Hits
    using hindex_type = uint32_t; //could be rolled back to TrackerTraits having the SoA with the relaxed uint32_t
    using PhiBinner = cms::alpakatools::HistoContainer<int16_t,
                                                     256,
                                                     -1, 
                                                     8 * sizeof(int16_t),
                                                     hindex_type,
                                                     TrackerTraits::numberOfLayers>; 
    using PhiBinnerStorageType = typename PhiBinner::index_type;
    using PhiBinnerView = typename PhiBinner::View;

    /// Hits in Tracks
    static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;
    static constexpr int32_t H = TrackerTraits::avgHitsPerTrack;
    using HitToTuple = caStructures::template HitToTupleT<TrackerTraits>;
    using HitContainer = caStructures::SequentialContainer;
    using HitToTupleView = typename HitToTuple::View;
    using TupleMultiplicity = caStructures::template TupleMultiplicityT<TrackerTraits>;
    using HitToCell = caStructures::GenericContainer;

    using GenericContainer = caStructures::GenericContainer;
    using GenericContainerStorage = typename GenericContainer::index_type;
    using GenericContainerView = typename GenericContainer::View;
    using DeviceGenericContainerBuffer = cms::alpakatools::device_buffer<Device, GenericContainer>;
    using DeviceGenericStorageBuffer = cms::alpakatools::device_buffer<Device, GenericContainerStorage[]>;
    using DeviceGenericOffsetsBuffer = cms::alpakatools::device_buffer<Device, GenericContainerOffsets[]>;

    using SequentialContainer = caStructures::SequentialContainer;
    using SequentialContainerStorage = typename SequentialContainer::index_type;
    using SequentialContainerView = typename SequentialContainer::View;
    using DeviceSequentialContainerBuffer = cms::alpakatools::device_buffer<Device, SequentialContainer>;
    using DeviceSequentialStorageBuffer = cms::alpakatools::device_buffer<Device, SequentialContainerStorage[]>;
    using DeviceSequentialOffsetsBuffer = cms::alpakatools::device_buffer<Device, SequentialContainerOffsets[]>;

    CAHitNtupletGeneratorKernels(Params const& params, uint32_t nHits, uint32_t offsetBPIX2, uint16_t nLayers, Queue& queue);
    ~CAHitNtupletGeneratorKernels() = default;

    GenericContainer const* tupleMultiplicity() const { return device_tupleMultiplicity_.data(); }
    SequentialContainer const* hitContainer() const { return device_hitContainer_.data(); }
    HitToCell const* hitToCell() const { return device_hitToCell_.data(); }

    void prepareHits(const HitsConstView& hh, const HitModulesConstView &mm, const ::reco::CALayersSoAConstView& ll, Queue& queue);

    void launchKernels(const HitsConstView& hh, uint32_t offsetBPIX2, uint16_t nLayers, TkSoAView& track_view, TkHitsSoAView& track_hits_view, const ::reco::CALayersSoAConstView& ca_layers, const ::reco::CACellsSoAConstView& ca_cells, Queue& queue);

    void classifyTuples(const HitsConstView& hh, TkSoAView& track_view, Queue& queue);

    void buildDoublets(const HitsConstView& hh, const ::reco::CACellsSoAConstView& cc, uint32_t offsetBPIX2, Queue& queue);

    static void printCounters();


  private:
    // params
    Params const& m_params;
    cms::alpakatools::device_buffer<Device, Counters> counters_;

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
    cms::alpakatools::device_buffer<Device, PhiBinner> device_hitPhiHist_;
    PhiBinnerView device_hitPhiView_;
    cms::alpakatools::device_buffer<Device, PhiBinnerStorageType[]> device_phiBinnerStorage_;
    cms::alpakatools::device_buffer<Device, hindex_type[]> device_layerStarts_;

    // Cells-> Neighbor Cells
    DeviceGenericContainerBuffer device_cellToNeighbors_;
    DeviceGenericStorageBuffer device_cellToNeighborsStorage_;
    DeviceGenericOffsetsBuffer device_cellToNeighborsOffsets_;
    GenericContainerView device_cellToNeighborsView_;

    // Tracks->Hits
    DeviceSequentialContainerBuffer device_hitContainer_;
    DeviceGenericStorageBuffer device_hitContainerStorage_;
    DeviceSequentialOffsetsBuffer device_hitContainerOffsets_;
    SequentialContainerView device_hitContainerView_;
    
    // No.Hits -> Track (Multiplicity)
    DeviceGenericContainerBuffer device_tupleMultiplicity_;
    DeviceGenericStorageBuffer device_tupleMultiplicityStorage_;
    DeviceGenericOffsetsBuffer device_tupleMultiplicityOffsets_;
    GenericContainerView device_tupleMultiplicityView_;

    cms::alpakatools::device_buffer<Device, CACell[]> device_theCells_;
    cms::alpakatools::device_buffer<Device, SimpleCell[]> device_simpleCells_;
    cms::alpakatools::device_buffer<Device, OuterHitOfCellContainer[]> device_isOuterHitOfCell_;
    cms::alpakatools::device_buffer<Device, OuterHitOfCell> isOuterHitOfCell_;
    cms::alpakatools::device_buffer<Device, CellNeighborsVector> device_theCellNeighbors_;
    cms::alpakatools::device_buffer<Device, CellTracksVector> device_theCellTracks_;
    cms::alpakatools::device_buffer<Device, unsigned char[]> cellStorage_;
    CellNeighbors* device_theCellNeighborsContainer_;
    CellTracks* device_theCellTracksContainer_;
    cms::alpakatools::device_buffer<Device, cms::alpakatools::AtomicPairCounter::DoubleWord[]> device_storage_;
    cms::alpakatools::AtomicPairCounter* device_hitTuple_apc_;
    cms::alpakatools::device_view<Device, uint32_t> device_nCells_;
    cms::alpakatools::device_view<Device, uint32_t> device_nTriplets_;

    CACoupleSoACollection deviceTriplets_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
