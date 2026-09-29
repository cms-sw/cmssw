#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h

#include <cmath>
#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

#include "RecoTracker/PixelSeeding/interface/CAHitsView.h"

namespace caStructures {

  using Quality = ::pixelTrack::Quality;

  //Configuration params common to all topologies, for the algorithms
  struct AlgoParams {
    // Container sizes
    float avgHitsPerTrack_;
    float avgCellsPerHit_;
    float avgCellsPerCell_;
    float avgTracksPerCell_;

    // Algorithm Parameters
    // NOTE: minHitsPerNtuplet_ is compared against the number of LAYERS in the ntuplet
    // (CACell::find_ntuplets); the name is kept because it is the name of the configuration
    // parameter set by the menus.
    uint16_t minHitsPerNtuplet_;
    uint16_t minHitsForSharingCut_;

    // Flags
    bool useRiemannFit_;
    // Enables the broken-line fit corrections (material partition and rigid-node guard, Karimaki Fisher
    // basis, pion 1/beta, trapezoid quadrature, 3x3 covariance blend; see the header comment of
    // RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h). Read only by the CA main fit.
    // Positional struct: keep this slot in the same
    // order as the makeCommonParams initializer in CAHitNtupletGenerator.cc.
    bool useFitCorrections_;
    bool fitNas4_;
    bool earlyFishbone_;
    bool lateFishbone_;
    bool onlySameLayersFishbone_;
    bool doStats_;
    bool doSharedHitCut_;
    bool dupPassThrough_;
    bool useSimpleTripletCleaner_;
    bool doTripletCleaner_;
    bool doFastDuplicateRemover_;
    bool doEarlyDuplicateRemover_;

    // Inline per-triplet DNN gate (Phase2OTStubs only): when enabled, an MLP with compile-time
    // weights (CATripletDNN.h) scores EVERY accepted triplet in Kernel_connect -- pixel-only
    // (nStubs==0, sentinel stub features) and stub-containing alike -- and rejects those below
    // tripletDNNThreshold_ (negative => the threshold compiled into the weight header).
    bool useTripletDNN_;
    float tripletDNNThreshold_;

    // Classify-embedded track classifier (Phase2OTStubs only): when enabled, an MLP with
    // compile-time weights (CATrackDNN.h) scores each fitted candidate in Kernel_classifyTracks
    // and its score REPLACES the chi2-based strict->tight promotion (trackDNNThreshold_
    // negative => the threshold compiled into the weight header).
    bool useTrackDNN_;
    float trackDNNThreshold_;

    // Device-memory allocation strategy (see CAHitNtupletGeneratorKernels)
    bool delayAllocations_;    // Defer cell-derived + hit->track buffers until their real size is known
    bool countDoubletsFirst_;  // Run a count-only doublet pass to size simpleCells/hitToCellStorage exactly

    // CA fast-duplicate / shared-hit parameter-cov gate width. Two tracks are declared duplicates when
    // every fitted param p satisfies dp^2 <= fastDupNSigma2_*(cov_i+cov_j) (Kernel_fastDuplicateRemover
    // and Kernel_rejectDuplicate, five-parameter compatibility check). A width in units of the fitted
    // covariance: lowering it tightens the gate (fewer merges), raising it merges more aggressively.
    // The Phase-1 specializations of those kernels keep their own hard-wired nSigma2Phase1 constant.
    float fastDupNSigma2_;
  };

  // Hits data formats
  using HitsView = ::reco::TrackingRecHitView;
  using HitModulesConstView = ::reco::HitModuleSoAConstView;
  using HitsConstView = ::reco::TrackingRecHitConstView;

  // MultiViews for hits and modules
  using ModulesMultiView = SoAConstMultiView<HitModulesConstView, 2>;
  using HitsMultiView = SoAConstMultiView<HitsConstView, 2>;

  // How a topology sees the hits inside the CA. Every topology but Phase2OTStubs reads its hit
  // collections through the upstream MultiViews; Phase2OTStubs reads the pixel rechits and the stubs
  // SoA through the CAHitsView facade, which presents the same element proxy and the same global hit
  // index space (see CAHitsView.h).
  template <typename TrackerTraits>
  struct HitsViewFor {
    using type = HitsMultiView;
    using modules_type = ModulesMultiView;
  };

  template <>
  struct HitsViewFor<::pixelTopology::Phase2OTStubs> {
    using type = CAHitsView;
    using modules_type = CAHitsView;
  };

  template <typename TrackerTraits>
  using HitsViewT = typename HitsViewFor<TrackerTraits>::type;

  template <typename TrackerTraits>
  using ModulesViewT = typename HitsViewFor<TrackerTraits>::modules_type;

  // True only for the facade, which carries the stubs SoA beside the pixel one.
  template <typename View>
  inline constexpr bool viewHasStubs = false;
  template <>
  inline constexpr bool viewHasStubs<CAHitsView> = true;

  // Everything the CA build hands to the generator: the hit view, the module-start view (the same
  // object as the hit view for the facade), and the two scalars the producer already knows.
  template <typename TrackerTraits>
  struct HitsInputT {
    HitsViewT<TrackerTraits> hits;
    ModulesViewT<TrackerTraits> modules;
    uint32_t nHits = 0;
    int32_t offsetBPIX2 = 0;
  };

  // Module start, as a global hit index, for CA module m. The MultiView holds one row per module of
  // the merged collection; the facade splices the pixel and the outer-tracker module blocks.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t moduleStartOf(ModulesMultiView const& mm, int32_t m) {
    return mm[m].moduleStart();
  }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t moduleStartOf(CAHitsView const& hh, int32_t m) {
    return hh.moduleStart(m);
  }

  // First stub index. The facade knows it as the number of pixel rechits; a MultiView topology has no
  // stubs at all, so it answers the "no stubs" sentinel (the stub code paths that read it are
  // compile-time gated on Phase2OTStubs, and int32_t(sentinel) < 0 disables the run-time ones).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t offsetStubsOf(HitsMultiView const&) {
    return std::numeric_limits<uint32_t>::max();
  }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t offsetStubsOf(CAHitsView const& hh) { return hh.offsetStubs(); }

  // Hit-classification predicates, by overload. The CAHitsView forms live in CAHitsView.h; a MultiView
  // topology carries no stubs and no outer-tracker entries, so every hit answers false.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isStub(HitsMultiView const&, int32_t) { return false; }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isOTEntry(HitsMultiView const&, int32_t) { return false; }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool hasBend(HitsMultiView const&, int32_t) { return false; }

  // Pixel cluster size in Y, for the doublet cuts, which are shared by every topology. The MultiView
  // reads the column directly; the facade reads it through the pixel element, which asserts that the
  // index is a pixel one (dSizeCut and clusterCut establish that before calling).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int16_t clusterSizeY(HitsMultiView const& hh, int32_t i) {
    return hh[i].clusterSizeY();
  }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int16_t clusterSizeY(CAHitsView const& hh, int32_t i) {
    return hh.pixel(i).clusterSizeY();
  }

  //Tracks data formats
  using TkSoAView = ::reco::TrackSoAView;
  using TkHitsSoAView = ::reco::TrackHitSoAView;
  using TkSoABlocksView = ::reco::TrackBlocksView;

  // Indices for hits, tracks and cells
  using hindex_type = uint32_t;
  using tindex_type = uint32_t;
  using cindex_type = uint32_t;

  // cellNeighbors storage encoding: bit 31 of each stored neighbor cell index
  // distinguishes layer-skipping (1) from non-layer-skipping (0) neighbors,
  // so the histogram needs a single bin per cell instead of two.
  inline constexpr uint32_t kSkipsLayerFlag = 0x80000000u;
  inline constexpr uint32_t kCellIndexMask = 0x7FFFFFFFu;

  using GenericContainer = cms::alpakatools::
      OneToManyAssocRandomAccess<hindex_type, cms::alpakatools::kDynamicSize, cms::alpakatools::kDynamicSize>;
  using GenericContainerStorage = typename GenericContainer::value_type;
  using GenericContainerOffsets = typename GenericContainer::Counter;
  using GenericContainerView = typename GenericContainer::View;

  using SequentialContainer = cms::alpakatools::
      OneToManyAssocSequential<hindex_type, cms::alpakatools::kDynamicSize, cms::alpakatools::kDynamicSize>;
  using SequentialContainerStorage = typename SequentialContainer::value_type;
  using SequentialContainerOffsets = typename SequentialContainer::Counter;
  using SequentialContainerView = typename SequentialContainer::View;

  template <typename TrackerTraits>
  using PhiBinnerT = cms::alpakatools::HistoContainer<int16_t,
                                                      256,
                                                      cms::alpakatools::kDynamicSize,
                                                      8 * sizeof(int16_t),
                                                      hindex_type,
                                                      TrackerTraits::numberOfLayers>;

  template <typename TrackerTraits>
  using CellNeighborsT =
      cms::alpakatools::VecArray<typename TrackerTraits::cindex_type, TrackerTraits::maxCellNeighbors>;

  template <typename TrackerTraits>
  using CellTracksT = cms::alpakatools::VecArray<tindex_type, TrackerTraits::maxCellTracks>;

  template <typename TrackerTraits>
  using CellNeighborsVectorT = cms::alpakatools::SimpleVector<CellNeighborsT<TrackerTraits>>;

  template <typename TrackerTraits>
  using CellTracksVectorT = cms::alpakatools::SimpleVector<CellTracksT<TrackerTraits>>;

  template <typename TrackerTraits>
  using OuterHitOfCellContainerT = cms::alpakatools::VecArray<uint32_t, TrackerTraits::maxCellsPerHit>;

  template <typename TrackerTraits>
  using TupleMultiplicityT = cms::alpakatools::
      OneToManyAssocRandomAccess<tindex_type, TrackerTraits::maxHitsOnTrack + 1, TrackerTraits::maxNumberOfTuples>;

  template <typename TrackerTraits>
  using HitContainerT =
      cms::alpakatools::OneToManyAssocSequential<uint32_t,
                                                 TrackerTraits::maxNumberOfTuples + 1,
                                                 TrackerTraits::avgHitsPerTrack * TrackerTraits::maxNumberOfTuples>;

  template <typename TrackerTraits>
  using HitToTupleT =
      cms::alpakatools::OneToManyAssocRandomAccess<tindex_type,
                                                   cms::alpakatools::kDynamicSize,
                                                   TrackerTraits::maxNumberOfTuples *
                                                       TrackerTraits::avgHitsPerTrack>;  // 3.5 should be enough

  template <typename TrackerTraits>
  using TuplesContainerT = cms::alpakatools::OneToManyAssocRandomAccess<typename TrackerTraits::hindex_type,
                                                                        TrackerTraits::maxNumberOfTuples,
                                                                        TrackerTraits::maxHitsForContainers>;

  template <typename TrackerTraits>
  struct OuterHitOfCellT {
    OuterHitOfCellContainerT<TrackerTraits>* container;
    int32_t offset;
    constexpr auto& operator[](int i) { return container[i - offset]; }
    constexpr auto const& operator[](int i) const { return container[i - offset]; }
  };

}  // namespace caStructures

// The CA kernels call isStub(hh, i) / isOTEntry(hh, i) / hasBend(hh, i) unqualified so the overload
// follows the hit view. ADL finds the caStructures ones for the CAHitsView facade, but not for a
// MultiView topology (whose associated namespaces are reco and the SoA template's), so the overload
// set is opened here once for every kernel of this backend namespace rather than in each kernel header.
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using caStructures::hasBend;
  using caStructures::isOTEntry;
  using caStructures::isStub;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h
