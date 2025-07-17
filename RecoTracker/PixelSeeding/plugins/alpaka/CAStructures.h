#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h

#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

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
    uint16_t minHitsPerNtuplet_;
    uint16_t minHitsForSharingCut_;
    float ptmin_;
    float hardCurvCut_;
    float cellZ0Cut_;
    float cellPtCut_;

    // Pixel Cluster Cut Params
    float dzdrFact_;  // from dz/dr to "DY"
    int16_t minYsizeB1_;
    int16_t minYsizeB2_;
    int16_t maxDYsize12_;
    int16_t maxDYsize_;
    int16_t maxDYPred_;

    // Flags
    bool useRiemannFit_;
    bool fitNas4_;
    bool earlyFishbone_;
    bool lateFishbone_;
    bool doStats_;
    bool doSharedHitCut_;
    bool dupPassThrough_;
    bool useSimpleTripletCleaner_;
  };

  // Hits data formats
  using HitsView = ::reco::TrackingRecHitView;
  using HitModulesConstView = ::reco::HitModuleSoAConstView;
  using HitsConstView = ::reco::TrackingRecHitConstView;

  //Tracks data formats
  using TkSoAView = ::reco::TrackSoAView;
  using TkHitsSoAView = ::reco::TrackHitSoAView;

  //Indices for hits, tracks and cells
  using hindex_type = uint32_t;
  using tindex_type = uint32_t;
  using cindex_type = uint32_t;

  using GenericContainer = cms::alpakatools::OneToManyAssocRandomAccess<hindex_type, -1, -1>;
  using GenericContainerStorage = typename GenericContainer::index_type;
  using GenericContainerOffsets = typename GenericContainer::Counter;
  using GenericContainerView = typename GenericContainer::View;

  using SequentialContainer = cms::alpakatools::OneToManyAssocSequential<hindex_type, -1, -1>;
  using SequentialContainerStorage = typename SequentialContainer::index_type;
  using SequentialContainerOffsets = typename SequentialContainer::Counter;
  using SequentialContainerView = typename SequentialContainer::View;

  template <typename TrackerTraits>
  using PhiBinnerT =
      cms::alpakatools::HistoContainer<int16_t, 256, -1, 8 * sizeof(int16_t), hindex_type, TrackerTraits::numberOfLayers>;

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
                                                   -1,
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

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h
