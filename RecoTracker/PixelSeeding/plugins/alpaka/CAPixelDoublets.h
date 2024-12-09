#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoublets_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoublets_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

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
        ALPAKA_ASSERT_ACC((*isOuterHitOfCell).container);

        for (auto i : cms::alpakatools::uniform_elements(acc, nHits - isOuterHitOfCell->offset))
          (*isOuterHitOfCell).container[i].reset();

        if (cms::alpakatools::once_per_grid(acc)) {
          cellNeighbors->construct(TrackerTraits::maxNumOfActiveDoublets, cellNeighborsContainer);
          cellTracks->construct(TrackerTraits::maxNumOfActiveDoublets, cellTracksContainer);
          [[maybe_unused]] auto i = cellNeighbors->extend(acc);
          ALPAKA_ASSERT_ACC(0 == i);
          (*cellNeighbors)[0].reset();
          i = cellTracks->extend(acc);
          ALPAKA_ASSERT_ACC(0 == i);
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
                                    // CACellT<TrackerTraits>* cells,
                                    CASimpleCell<TrackerTraits>* cells,
                                    uint32_t* nCells,
                                    // CellNeighborsVector<TrackerTraits>* cellNeighbors,
                                    // CellTracksVector<TrackerTraits>* cellTracks,
                                    HitsConstView hh,
                                    ::reco::CACellsSoAConstView cc,
                                    uint32_t const* __restrict__ offsets,
                                    PhiBinner<TrackerTraits>* phiBinner,
                                    // OuterHitOfCell<TrackerTraits>* isOuterHitOfCell,
                                    HitToCell* outerHitHisto,
                                    AlgoParams const& params) const {
        doubletsFromHisto<TrackerTraits>(
            acc, cells, nCells, hh, cc, offsets, phiBinner, outerHitHisto, params);
      }
    };

    template <typename TrackerTraits>
    class FillDoubletsHisto {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    CASimpleCell<TrackerTraits> const* __restrict__ cells,
                                    uint32_t* nCells, //could be size
                                    uint32_t offsetBPIX2,
                                    HitToCell* outerHitHisto) const {
        for (auto cellIndex : cms::alpakatools::uniform_elements(acc, *nCells))
        {
          printf("outerHitHisto;%d;%d\n",cellIndex,cells[cellIndex].outer_hit_id());
          outerHitHisto->fill(acc,cells[cellIndex].outer_hit_id()-offsetBPIX2,cellIndex);
        }
      }
    };

  }  // namespace caPixelDoublets

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoublets_h
