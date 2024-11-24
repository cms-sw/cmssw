#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h

#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

namespace caStructures {


  using Quality = ::pixelTrack::Quality;


  //Configuration params common to all topologies, for the algorithms
  struct AlgoParams {

    uint32_t maxNumberOfDoublets_;
    uint32_t minHitsPerNtuplet_;
    uint32_t minHitsForSharingCut_;

    float ptmin_;
    float hardCurvCut_;
    bool useRiemannFit_;
    bool fitNas4_;
    bool earlyFishbone_;
    bool lateFishbone_;
    bool doStats_;
    bool doSharedHitCut_;
    bool dupPassThrough_;
    bool useSimpleTripletCleaner_;

    // uint32_t maxNumberOfTriplets_ = 1; // === maxDoublets * avgCellPerCell
    uint32_t maxNumberOfTuples_ = 1;
    uint8_t avgHitsPerTrack_ = 1;
    uint8_t avgCellPerHit_ = 1;
    uint8_t avgCellPerCell_ = 1;
    uint8_t avgTrackPerCell_ = 1;
    uint8_t avgNeighborPerCell_ = 1;
    // bool idealConditions_;
    //move back idealConditions here
  };

  // Hits data formats
  using HitsView = ::reco::TrackingRecHitView;
  using HitModulesConstView = ::reco::HitModuleSoAConstView;
  using HitsConstView = ::reco::TrackingRecHitConstView;

  //Tracks data formats
  using TkSoAView = ::reco::TrackSoAView;
  using TkHitsSoAView = ::reco::TrackHitSoAView;

  //Indices for hits and tracks
  using hindex_type = uint32_t; // TrackerTraits::hindex_type
  using tindex_type = uint32_t; // TrackerTraits::tindex_type
  using cindex_type = uint32_t;

  // template <typename TrackerTraits>
  // struct CAContainers
  // {
  //   //Max constants
  //   static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;
  //   static constexpr int32_t H = TrackerTraits::avgHitsPerTrack;
  //   static constexpr uint32_t CT = TrackerTraits::maxCellTracks;
  //   static constexpr uint32_t CN = TrackerTraits::maxCellNeighbors;

  //   using CellNeighbors = cms::alpakatools::VecArray<typename TrackerTraits::cindex_type, TrackerTraits::maxCellNeighbors>;
  // }

  using GenericContainer = cms::alpakatools::OneToManyAssocRandomAccess<hindex_type, -1, -1>;
  using GenericContainerStorage = uint32_t;//typename GenericContainer::index_type;
  using GenericContainerOffsets = uint32_t;//typename GenericContainer::Counter;
  using GenericContainerView = typename GenericContainer::View;

  using SequentialContainer = cms::alpakatools::OneToManyAssocSequential<hindex_type, -1, -1>;
  using SequentialContainerStorage = uint32_t;//typename SequentialContainer::index_type;
  using SequentialContainerOffsets = uint32_t;//typename SequentialContainer::Counter;
  using SequentialContainerView = typename SequentialContainer::View;

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
  using TupleMultiplicityT = cms::alpakatools::OneToManyAssocRandomAccess<tindex_type,
                                                                          TrackerTraits::maxHitsOnTrack + 1,
                                                                          TrackerTraits::maxNumberOfTuples>;
  
  template <typename TrackerTraits>
  using HitContainerT = cms::alpakatools::OneToManyAssocSequential<uint32_t, TrackerTraits::maxNumberOfTuples + 1, 
                                        TrackerTraits::avgHitsPerTrack * TrackerTraits::maxNumberOfTuples>;
                                          
  template <typename TrackerTraits>
  using HitToTupleT =
      cms::alpakatools::OneToManyAssocRandomAccess<tindex_type,
                                                   -1,
                                                   TrackerTraits::maxNumberOfTuples*TrackerTraits::avgHitsPerTrack>;  // 3.5 should be enough
  
  

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
