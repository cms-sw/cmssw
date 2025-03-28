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
    class GetDoubletsFromHisto {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      // #ifdef __CUDACC__
      //       __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)  // TODO: Alapakafy
      // #endif
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t maxNumOfDoublets,
                                    CASimpleCell<TrackerTraits>* cells,
                                    uint32_t* nCells,
                                    HitsConstView hh,
                                    ::reco::CAGraphSoAConstView cc,
                                    ::reco::CALayersSoAConstView ll,
                                    uint32_t const* __restrict__ offsets,
                                    PhiBinner<TrackerTraits> const* phiBinner,
                                    HitToCell* outerHitHisto,
                                    AlgoParams const& params) const {
        doubletsFromHisto<TrackerTraits>(
            acc, maxNumOfDoublets, cells, nCells, hh, cc, ll, offsets, phiBinner, outerHitHisto, params);
      }
    };

    template <typename TrackerTraits>
    class FillDoubletsHisto {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    CASimpleCell<TrackerTraits> const* __restrict__ cells,
                                    uint32_t* nCells,
                                    uint32_t offsetBPIX2,
                                    HitToCell* outerHitHisto) const {
        for (auto cellIndex : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef DOUBLETS_DEBUG
          printf("outerHitHisto;%d;%d\n", cellIndex, cells[cellIndex].outer_hit_id());
#endif
          outerHitHisto->fill(acc, cells[cellIndex].outer_hit_id() - offsetBPIX2, cellIndex);
        }
      }
    };

  }  // namespace caPixelDoublets

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoublets_h
