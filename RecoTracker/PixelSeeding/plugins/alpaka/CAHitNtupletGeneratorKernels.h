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
// #include "RecoTracker/PixelSeeding/interface/alpaka/CACoupleSoACollection.h"

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
    using HitContainer = cms::alpakatools::OneToManyAssocSequential<hindex_type, S + 1, H * S>;
    using HitToTupleView = typename HitToTuple::View;
    using TupleMultiplicity = caStructures::template TupleMultiplicityT<TrackerTraits>;
    
    /// Cells
    using GenericContainer = caStructures::GenericContainer;
    using GenericContainerStorage = typename GenericContainer::index_type;
    using GenericContainerView = typename GenericContainer::View;

    CAHitNtupletGeneratorKernels(Params const& params, const HitsConstView &hh, uint16_t nLayers, Queue& queue);
    ~CAHitNtupletGeneratorKernels() = default;

    TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_.data(); }
    HitContainer const* hitContainer() const { return device_hitContainer_.data(); }

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
    // cms::alpakatools::device_buffer<Device, HitToTuple> device_hitToTuple_;
    // cms::alpakatools::device_buffer<Device, uint32_t[]> device_hitToTupleStorage_;

    cms::alpakatools::device_buffer<Device, GenericContainer> device_hitToTuple_;
    cms::alpakatools::device_buffer<Device, GenericContainerStorage[]> device_hitToTupleStorage_;
    cms::alpakatools::device_buffer<Device, GenericContainerOffsets[]> device_hitToTupleOffsets_;

    // Hits 
    cms::alpakatools::device_buffer<Device, PhiBinner> device_hitPhiHist_;
    PhiBinnerView device_hitPhiView_;
    cms::alpakatools::device_buffer<Device, PhiBinnerStorageType[]> device_phiBinnerStorage_;
    cms::alpakatools::device_buffer<Device, hindex_type[]> device_layerStarts_;

    // Tracks->Hits
    cms::alpakatools::device_buffer<Device, HitContainer> device_hitContainer_;
    
    // GenericContainerView device_hitToTupleView_;
    GenericContainerView device_hitToTupleView_;
    cms::alpakatools::device_buffer<Device, TupleMultiplicity> device_tupleMultiplicity_;
    cms::alpakatools::device_buffer<Device, CACell[]> device_theCells_;
    cms::alpakatools::device_buffer<Device, OuterHitOfCellContainer[]> device_isOuterHitOfCell_;
    cms::alpakatools::device_buffer<Device, OuterHitOfCell> isOuterHitOfCell_;
    cms::alpakatools::device_buffer<Device, CellNeighborsVector> device_theCellNeighbors_;
    cms::alpakatools::device_buffer<Device, CellTracksVector> device_theCellTracks_;
    cms::alpakatools::device_buffer<Device, unsigned char[]> cellStorage_;
    CellNeighbors* device_theCellNeighborsContainer_;
    CellTracks* device_theCellTracksContainer_;
    cms::alpakatools::device_buffer<Device, cms::alpakatools::AtomicPairCounter::DoubleWord[]> device_storage_;
    cms::alpakatools::AtomicPairCounter* device_hitTuple_apc_;
    cms::alpakatools::AtomicPairCounter* device_hitToTuple_apc_;
    cms::alpakatools::device_view<Device, uint32_t> device_nCells_;
    // cms::alpakatools::host_buffer<uint32_t> host_nCells_;
    // Hit -> Cells
    // cms::alpakatools::device_buffer<Device, GenericContainer> device_hitToCell_;
    // GenericContainerView device_hitToCellView_;
    // cms::alpakatools::device_buffer<Device, GenericContainerStorage[]> device_hitToCellOffsets_;

    // cms::alpakatools::device_buffer<Device, GenericContainerStorage[]> device_hitToCellStorage_;
    // CACoupleSoACollection 
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
