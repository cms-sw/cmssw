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

#include "CACell.h"
#include "CAPixelDoublets.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace caHitNtupletGenerator {

    //Configuration params common to all topologies, for the algorithms
    struct AlgoParams {
      const uint32_t minHitsForSharingCut_;
      const bool useRiemannFit_;
      const bool fitNas4_;
      const bool includeJumpingForwardDoublets_;
      const bool earlyFishbone_;
      const bool lateFishbone_;
      const bool doStats_;
      const bool doSharedHitCut_;
      const bool dupPassThrough_;
      const bool useSimpleTripletCleaner_;
      const bool useRemovers_;
    };

    //CAParams
    struct CACommon {
      const uint32_t maxNumberOfDoublets_;
      const uint32_t minHitsPerNtuplet_;
      const float ptmin_;
      const float CAThetaCutBarrel_;
      const float CAThetaCutForward_;
      const float hardCurvCut_;
      const float dcaCutInnerTriplet_;
      const float dcaCutOuterTriplet_;
      const float CAThetaCutBarrelPixelBarrelStrip_;
      const float CAThetaCutBarrelPixelForwardStrip_;
      const float CAThetaCutBarrelStripForwardStrip_;
      const float CAThetaCutBarrelStrip_;
      const float CAThetaCutDefault_;
      const float dcaCutInnerTripletPixelStrip_;
      const float dcaCutOuterTripletPixelStrip_;
      const float dcaCutTripletStrip_;
      const float dcaCutTripletDefault_;
    };

    template <typename TrackerTraits, typename Enable = void>
    struct CAParamsT : public CACommon {
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool startingLayerPair(int16_t pid) const { return false; };
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool startAt0(int16_t pid) const { return false; };
    };

    template <typename TrackerTraits>
    struct CAParamsT<TrackerTraits, pixelTopology::isPhase1Topology<TrackerTraits>> : public CACommon {
      /// Is is a starting layer pair?
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool startingLayerPair(int16_t pid) const {
        if constexpr (std::is_same_v<TrackerTraits, pixelTopology::Phase1Strip>) {
          return (pid < 12 || pid == 32 || pid == 33 || pid == 29 || pid == 27 || pid == 18 || pid == 37);
        } else {
          return minHitsPerNtuplet_ > 3 ? pid < 3 : pid < 8 || pid > 12;
        }
      }

      /// Is this a pair with inner == 0?
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool startAt0(int16_t pid) const {
        if constexpr (std::is_same_v<TrackerTraits, pixelTopology::Phase1Strip>) {
          assert((pixelTopology::Phase1Strip::layerPairs[pid * 2] == 0) ==
                 (pid < 3 || pid == 8 || pid == 10 || pid == 11 ||
                  pid == 29));  // to be 100% sure it's working, may be removed
          return pixelTopology::Phase1Strip::layerPairs[pid * 2] == 0;
        } else {
          assert((pixelTopology::Phase1::layerPairs[pid * 2] == 0) ==
                 (pid < 3 || pid == 13 || pid == 15 || pid == 16));  // to be 100% sure it's working, may be removed
          return pixelTopology::Phase1::layerPairs[pid * 2] == 0;
        }
      }
    };

    template <typename TrackerTraits>
    struct CAParamsT<TrackerTraits, pixelTopology::isPhase2Topology<TrackerTraits>> : public CACommon {
      const bool includeFarForwards_;
      /// Is is a starting layer pair?
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool startingLayerPair(int16_t pid) const {
        return pid < 33;  // in principle one could remove 5,6,7 23, 28 and 29
      }

      /// Is this a pair with inner == 0
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool startAt0(int16_t pid) const {
        ALPAKA_ASSERT_ACC((pixelTopology::Phase2::layerPairs[pid * 2] == 0) == ((pid < 3) | (pid >= 23 && pid < 28)));
        return pixelTopology::Phase2::layerPairs[pid * 2] == 0;
      }
    };

    //Full list of params = algo params + ca params + cell params + quality cuts
    //Generic template
    template <typename TrackerTraits, typename Enable = void>
    struct ParamsT : public AlgoParams {
      // one should define the params for its own pixelTopology
      // not defining anything here
      inline uint32_t nPairs() const { return 0; }
    };

    template <typename TrackerTraits>
    struct ParamsT<TrackerTraits, pixelTopology::isPhase1Topology<TrackerTraits>> : public AlgoParams {
      using TT = TrackerTraits;
      using QualityCuts = ::pixelTrack::QualityCutsT<TT>;  //track quality cuts
      using CellCuts = caPixelDoublets::CellCutsT<TT>;     //cell building cuts
      using CAParams = CAParamsT<TT>;                      //params to be used on device

      ParamsT(AlgoParams const& commonCuts,
              CellCuts const& cellCuts,
              QualityCuts const& cutsCuts,
              CAParams const& caParams)
          : AlgoParams(commonCuts), cellCuts_(cellCuts), qualityCuts_(cutsCuts), caParams_(caParams) {}

      const CellCuts cellCuts_;
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
      const CAParams caParams_;
      /// Compute the number of pairs
      inline uint32_t nPairs() const {
        // take all layer pairs into account
        uint32_t nActualPairs = TT::nPairs;
        if (not includeJumpingForwardDoublets_) {
          // exclude forward "jumping" layer pairs
          nActualPairs = TT::nPairsForTriplets;
        }
        if (caParams_.minHitsPerNtuplet_ > 3) {
          // for quadruplets, exclude all "jumping" layer pairs
          nActualPairs = TT::nPairsForQuadruplets;
        }

        return nActualPairs;
      }

    };  // Params Phase1

    template <typename TrackerTraits>
    struct ParamsT<TrackerTraits, pixelTopology::isPhase2Topology<TrackerTraits>> : public AlgoParams {
      using TT = TrackerTraits;
      using QualityCuts = ::pixelTrack::QualityCutsT<TT>;
      using CellCuts = caPixelDoublets::CellCutsT<TT>;
      using CAParams = CAParamsT<TT>;

      ParamsT(AlgoParams const& commonCuts,
              CellCuts const& cellCuts,
              QualityCuts const& qualityCuts,
              CAParams const& caParams)
          : AlgoParams(commonCuts), cellCuts_(cellCuts), qualityCuts_(qualityCuts), caParams_(caParams) {}

      // quality cuts
      const CellCuts cellCuts_;
      const QualityCuts qualityCuts_{5.0f, /*chi2*/ 0.9f, /* pT in Gev*/ 0.4f, /*zip in cm*/ 12.0f /*tip in cm*/};
      const CAParams caParams_;

      inline uint32_t nPairs() const {
        // take all layer pairs into account
        uint32_t nActualPairs = TT::nPairsMinimal;
        if (caParams_.includeFarForwards_) {
          // considera far forwards (> 11 & > 23)
          nActualPairs = TT::nPairsFarForwards;
        }
        if (includeJumpingForwardDoublets_) {
          // include jumping forwards
          nActualPairs = TT::nPairs;
        }

        return nActualPairs;
      }

    };  // Params Phase1

    // counters
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

    using Quality = ::pixelTrack::Quality;

  }  // namespace caHitNtupletGenerator

  template <typename TTTraits>
  class CAHitNtupletGeneratorKernels {
  public:
    using TrackerTraits = TTTraits;
    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;
    using CellCuts = caPixelDoublets::CellCutsT<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using CAParams = caHitNtupletGenerator::CAParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;

    using HitsView = TrackingRecHitSoAView<TrackerTraits>;
    using HitsConstView = TrackingRecHitSoAConstView<TrackerTraits>;
    using TkSoAView = reco::TrackSoAView<TrackerTraits>;

    using HitToTuple = caStructures::template HitToTupleT<TrackerTraits>;
    using TupleMultiplicity = caStructures::template TupleMultiplicityT<TrackerTraits>;
    struct Testttt {
      TupleMultiplicity tm;
    };
    using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
    using CellNeighbors = caStructures::CellNeighborsT<TrackerTraits>;
    using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
    using CellTracks = caStructures::CellTracksT<TrackerTraits>;
    using OuterHitOfCellContainer = caStructures::OuterHitOfCellContainerT<TrackerTraits>;
    using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

    using CACell = CACellT<TrackerTraits>;

    using Quality = ::pixelTrack::Quality;
    using HitContainer = typename reco::TrackSoA<TrackerTraits>::HitContainer;

    CAHitNtupletGeneratorKernels(Params const& params, uint32_t nhits, uint32_t offsetBPIX2, Queue& queue);
    ~CAHitNtupletGeneratorKernels() = default;

    TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_.data(); }

    void launchKernels(const HitsConstView& hh, uint32_t offsetBPIX2, TkSoAView& track_view, Queue& queue);

    void classifyTuples(const HitsConstView& hh, TkSoAView& track_view, Queue& queue);

    void buildDoublets(const HitsConstView& hh, uint32_t offsetBPIX2, Queue& queue);

    static void printCounters();

  private:
    // params
    Params const& m_params;
    cms::alpakatools::device_buffer<Device, Counters> counters_;

    // workspace
    cms::alpakatools::device_buffer<Device, HitToTuple> device_hitToTuple_;
    cms::alpakatools::device_buffer<Device, uint32_t[]> device_hitToTupleStorage_;
    typename HitToTuple::View device_hitToTupleView_;
    cms::alpakatools::device_buffer<Device, TupleMultiplicity> device_tupleMultiplicity_;
    cms::alpakatools::device_buffer<Device, CACell[]> device_theCells_;
    cms::alpakatools::device_buffer<Device, OuterHitOfCellContainer[]> device_isOuterHitOfCell_;
    cms::alpakatools::device_buffer<Device, OuterHitOfCell> isOuterHitOfCell_;
    cms::alpakatools::device_buffer<Device, CellNeighborsVector> device_theCellNeighbors_;
    cms::alpakatools::device_buffer<Device, CellTracksVector> device_theCellTracks_;
    cms::alpakatools::device_buffer<Device, unsigned char[]> cellStorage_;
    cms::alpakatools::device_buffer<Device, CellCuts> device_cellCuts_;
    CellNeighbors* device_theCellNeighborsContainer_;
    CellTracks* device_theCellTracksContainer_;
    cms::alpakatools::device_buffer<Device, cms::alpakatools::AtomicPairCounter::DoubleWord[]> device_storage_;
    cms::alpakatools::AtomicPairCounter* device_hitTuple_apc_;
    cms::alpakatools::AtomicPairCounter* device_hitToTuple_apc_;
    cms::alpakatools::device_view<Device, uint32_t> device_nCells_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
