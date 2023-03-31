#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

// #define GPU_DEBUG

#include "GPUCACell.h"
#include "gpuPixelDoublets.h"

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"

// #define DUMP_GPU_TK_TUPLES

namespace caHitNtupletGenerator {

  //Configuration params common to all topologies, for the algorithms
  struct AlgoParams {
    const bool onGPU_;
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
  };

  //CAParams
  struct CACommon {
    const uint32_t minHitsPerNtuplet_;
    const float ptmin_;
    const float CAThetaCutBarrel_;
    const float CAThetaCutForward_;
    const float hardCurvCut_;
    const float dcaCutInnerTriplet_;
    const float dcaCutOuterTriplet_;
  };

  template <typename TrackerTraits, typename Enable = void>
  struct CAParamsT : public CACommon {
    __device__ __forceinline__ bool startingLayerPair(int16_t pid) const { return false; };
    __device__ __forceinline__ bool startAt0(int16_t pid) const { return false; };
  };

  template <typename TrackerTraits>
  struct CAParamsT<TrackerTraits, pixelTopology::isPhase1Topology<TrackerTraits>> : public CACommon {
    /// Is is a starting layer pair?
    __device__ __forceinline__ bool startingLayerPair(int16_t pid) const {
      return minHitsPerNtuplet_ > 3 ? pid < 3 : pid < 8 || pid > 12;
    }

    /// Is this a pair with inner == 0?
    __device__ __forceinline__ bool startAt0(int16_t pid) const {
      assert((pixelTopology::Phase1::layerPairs[pid * 2] == 0) ==
             (pid < 3 || pid == 13 || pid == 15 || pid == 16));  // to be 100% sure it's working, may be removed
      return pixelTopology::Phase1::layerPairs[pid * 2] == 0;
    }
  };

  template <typename TrackerTraits>
  struct CAParamsT<TrackerTraits, pixelTopology::isPhase2Topology<TrackerTraits>> : public CACommon {
    const bool includeFarForwards_;
    /// Is is a starting layer pair?
    __device__ __forceinline__ bool startingLayerPair(int16_t pid) const {
      return pid < 33;  // in principle one could remove 5,6,7 23, 28 and 29
    }

    /// Is this a pair with inner == 0
    __device__ __forceinline__ bool startAt0(int16_t pid) const {
      assert((pixelTopology::Phase2::layerPairs[pid * 2] == 0) == ((pid < 3) | (pid >= 23 && pid < 28)));
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
    using QualityCuts = pixelTrack::QualityCutsT<TT>;  //track quality cuts
    using CellCuts = gpuPixelDoublets::CellCutsT<TT>;  //cell building cuts
    using CAParams = CAParamsT<TT>;                    //params to be used on device

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
    using QualityCuts = pixelTrack::QualityCutsT<TT>;
    using CellCuts = gpuPixelDoublets::CellCutsT<TT>;
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

  using Quality = pixelTrack::Quality;

}  // namespace caHitNtupletGenerator

template <typename TTraits, typename TTTraits>
class CAHitNtupletGeneratorKernels {
public:
  using Traits = TTraits;
  using TrackerTraits = TTTraits;
  using QualityCuts = pixelTrack::QualityCutsT<TrackerTraits>;
  using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
  using CAParams = caHitNtupletGenerator::CAParamsT<TrackerTraits>;
  using Counters = caHitNtupletGenerator::Counters;

  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using HitsView = TrackingRecHitSoAView<TrackerTraits>;
  using HitsConstView = TrackingRecHitSoAConstView<TrackerTraits>;
  using TkSoAView = TrackSoAView<TrackerTraits>;

  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  using CellNeighbors = caStructures::CellNeighborsT<TrackerTraits>;
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  using CellTracks = caStructures::CellTracksT<TrackerTraits>;
  using OuterHitOfCellContainer = caStructures::OuterHitOfCellContainerT<TrackerTraits>;
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

  using CACell = GPUCACellT<TrackerTraits>;

  using Quality = pixelTrack::Quality;
  using HitContainer = typename TrackSoA<TrackerTraits>::HitContainer;

  CAHitNtupletGeneratorKernels(Params const& params)
      : params_(params), paramsMaxDoubletes3Quarters_(3 * params.cellCuts_.maxNumberOfDoublets_ / 4) {}

  ~CAHitNtupletGeneratorKernels() = default;

  TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_.get(); }

  void launchKernels(const HitsConstView& hh, TkSoAView& track_view, cudaStream_t cudaStream);

  void classifyTuples(const HitsConstView& hh, TkSoAView& track_view, cudaStream_t cudaStream);

  void buildDoublets(const HitsConstView& hh, cudaStream_t stream);
  void allocateOnGPU(int32_t nHits, cudaStream_t stream);
  void cleanup(cudaStream_t cudaStream);

  static void printCounters(Counters const* counters);
  void setCounters(Counters* counters) { counters_ = counters; }

protected:
  Counters* counters_ = nullptr;
  // workspace
  unique_ptr<unsigned char[]> cellStorage_;
  unique_ptr<CellNeighborsVector> device_theCellNeighbors_;
  CellNeighbors* device_theCellNeighborsContainer_;
  unique_ptr<CellTracksVector> device_theCellTracks_;
  CellTracks* device_theCellTracksContainer_;

  unique_ptr<CACell[]> device_theCells_;
  unique_ptr<OuterHitOfCellContainer[]> device_isOuterHitOfCell_;
  OuterHitOfCell isOuterHitOfCell_;
  uint32_t* device_nCells_ = nullptr;

  unique_ptr<HitToTuple> device_hitToTuple_;
  unique_ptr<uint32_t[]> device_hitToTupleStorage_;
  typename HitToTuple::View hitToTupleView_;

  cms::cuda::AtomicPairCounter* device_hitToTuple_apc_ = nullptr;

  cms::cuda::AtomicPairCounter* device_hitTuple_apc_ = nullptr;

  unique_ptr<TupleMultiplicity> device_tupleMultiplicity_;

  unique_ptr<cms::cuda::AtomicPairCounter::c_type[]> device_storage_;

  // params
  Params params_;
  /// Intermediate result avoiding repeated computations.
  const uint32_t paramsMaxDoubletes3Quarters_;
  /// Compute the number of doublet blocks for block size
  inline uint32_t nDoubletBlocks(uint32_t blockSize) {
    // We want (3 * params_.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize, but first part is pre-computed.
    return (paramsMaxDoubletes3Quarters_ + blockSize - 1) / blockSize;
  }

  /// Compute the number of quadruplet blocks for block size
  inline uint32_t nQuadrupletBlocks(uint32_t blockSize) {
    // pixelTopology::maxNumberOfQuadruplets is a constexpr, so the compiler will pre compute the 3*max/4
    return (3 * TrackerTraits::maxNumberOfQuadruplets / 4 + blockSize - 1) / blockSize;
  }
};

template <typename TrackerTraits>
class CAHitNtupletGeneratorKernelsGPU : public CAHitNtupletGeneratorKernels<cms::cudacompat::GPUTraits, TrackerTraits> {
  using CAHitNtupletGeneratorKernels<cms::cudacompat::GPUTraits, TrackerTraits>::CAHitNtupletGeneratorKernels;

  using Counters = caHitNtupletGenerator::Counters;
  using CAParams = caHitNtupletGenerator::CAParamsT<TrackerTraits>;

  using HitContainer = typename TrackSoA<TrackerTraits>::HitContainer;

  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

  using HitsConstView = TrackingRecHitSoAConstView<TrackerTraits>;
  using TkSoAView = TrackSoAView<TrackerTraits>;

public:
  void launchKernels(const HitsConstView& hh, TkSoAView& track_view, cudaStream_t cudaStream);
  void classifyTuples(const HitsConstView& hh, TkSoAView& track_view, cudaStream_t cudaStream);
  void buildDoublets(const HitsConstView& hh, int32_t offsetBPIX2, cudaStream_t stream);
  void allocateOnGPU(int32_t nHits, cudaStream_t stream);
  static void printCounters(Counters const* counters);
};

template <typename TrackerTraits>
class CAHitNtupletGeneratorKernelsCPU : public CAHitNtupletGeneratorKernels<cms::cudacompat::CPUTraits, TrackerTraits> {
  using CAHitNtupletGeneratorKernels<cms::cudacompat::CPUTraits, TrackerTraits>::CAHitNtupletGeneratorKernels;

  using Counters = caHitNtupletGenerator::Counters;
  using CAParams = caHitNtupletGenerator::CAParamsT<TrackerTraits>;

  using HitContainer = typename TrackSoA<TrackerTraits>::HitContainer;

  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

  using HitsConstView = TrackingRecHitSoAConstView<TrackerTraits>;
  using TkSoAView = TrackSoAView<TrackerTraits>;

public:
  void launchKernels(const HitsConstView& hh, TkSoAView& track_view, cudaStream_t cudaStream);
  void classifyTuples(const HitsConstView& hh, TkSoAView& track_view, cudaStream_t cudaStream);
  void buildDoublets(const HitsConstView& hh, int32_t offsetBPIX2, cudaStream_t stream);
  void allocateOnGPU(int32_t nHits, cudaStream_t stream);
  static void printCounters(Counters const* counters);
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
