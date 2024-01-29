#ifndef RecoPixelVertexing_PixelTriplets_alpaka_CAPixelDoublets_h
#define RecoPixelVertexing_PixelTriplets_alpaka_CAPixelDoublets_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "CAPixelDoubletsAlgos.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace alpaka;
  using namespace cms::alpakatools;
  namespace caPixelDoublets {

    template <typename TrackerTraits>
    class InitDoublets {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    OuterHitOfCell<TrackerTraits>* isOuterHitOfCell,
                                    int nHits,
                                    CellNeighborsVector<TrackerTraits>* cellNeighbors,
                                    CellNeighbors<TrackerTraits>* cellNeighborsContainer,
                                    CellTracksVector<TrackerTraits>* cellTracks,
                                    CellTracks<TrackerTraits>* cellTracksContainer) const {
        ALPAKA_ASSERT_OFFLOAD((*isOuterHitOfCell).container);

        for (auto i : cms::alpakatools::elements_with_stride(acc, nHits))
          (*isOuterHitOfCell).container[i].reset();

        if (cms::alpakatools::once_per_grid(acc)) {
          cellNeighbors->construct(TrackerTraits::maxNumOfActiveDoublets, cellNeighborsContainer);
          cellTracks->construct(TrackerTraits::maxNumOfActiveDoublets, cellTracksContainer);
          [[maybe_unused]] auto i = cellNeighbors->extend(acc);
          ALPAKA_ASSERT_OFFLOAD(0 == i);
          (*cellNeighbors)[0].reset();
          i = cellTracks->extend(acc);
          ALPAKA_ASSERT_OFFLOAD(0 == i);
          (*cellTracks)[0].reset();
        }
      }
    };

    // Not used for the moment, see below.
    //constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
    //constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

    template <typename TrackerTraits>
    class GetDoubletsFromHisto {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      // #ifdef __CUDACC__
      //       __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)  // TODO: Alapakify
      // #endif
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    CACellT<TrackerTraits>* cells,
                                    uint32_t* nCells,
                                    CellNeighborsVector<TrackerTraits>* cellNeighbors,
                                    CellTracksVector<TrackerTraits>* cellTracks,
                                    HitsConstView<TrackerTraits> hh,
                                    OuterHitOfCell<TrackerTraits>* isOuterHitOfCell,
                                    uint32_t nActualPairs,
                                    const uint32_t maxNumOfDoublets,
                                    CellCutsT<TrackerTraits> cuts) const {
        doubletsFromHisto<TrackerTraits>(
            acc, nActualPairs, maxNumOfDoublets, cells, nCells, cellNeighbors, cellTracks, hh, *isOuterHitOfCell, cuts);
      }
    };
  }  // namespace caPixelDoublets
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAPixelDoublets_h
