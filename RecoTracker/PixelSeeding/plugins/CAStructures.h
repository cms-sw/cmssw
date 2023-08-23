#ifndef RecoTracker_PixelSeeding_plugins_CAStructures_h
#define RecoTracker_PixelSeeding_plugins_CAStructures_h

#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

namespace caStructures {

  // types
  // using typename TrackerTraits::hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  // using typename TrackerTraits::tindex_type = uint32_t;  // for tuples
  // using typename TrackerTraits::cindex_type = uint32_t;  // for cells

  template <typename TrackerTraits>
  using CellNeighborsT = cms::cuda::VecArray<typename TrackerTraits::cindex_type, TrackerTraits::maxCellNeighbors>;

  template <typename TrackerTraits>
  using CellTracksT = cms::cuda::VecArray<typename TrackerTraits::tindex_type, TrackerTraits::maxCellTracks>;

  template <typename TrackerTraits>
  using CellNeighborsVectorT = cms::cuda::SimpleVector<CellNeighborsT<TrackerTraits>>;

  template <typename TrackerTraits>
  using CellTracksVectorT = cms::cuda::SimpleVector<CellTracksT<TrackerTraits>>;

  template <typename TrackerTraits>
  using OuterHitOfCellContainerT = cms::cuda::VecArray<uint32_t, TrackerTraits::maxCellsPerHit>;

  template <typename TrackerTraits>
  using TupleMultiplicityT = cms::cuda::OneToManyAssoc<typename TrackerTraits::tindex_type,
                                                       TrackerTraits::maxHitsOnTrack + 1,
                                                       TrackerTraits::maxNumberOfTuples>;

  template <typename TrackerTraits>
  using HitToTupleT = cms::cuda::OneToManyAssoc<typename TrackerTraits::tindex_type,
                                                -1,
                                                TrackerTraits::maxHitsForContainers>;  // 3.5 should be enough

  template <typename TrackerTraits>
  using TuplesContainerT = cms::cuda::OneToManyAssoc<typename TrackerTraits::hindex_type,
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

#endif
