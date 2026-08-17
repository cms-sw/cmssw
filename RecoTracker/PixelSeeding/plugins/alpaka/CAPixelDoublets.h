#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoublets_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoublets_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CAPixelDoubletsAlgos.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets {

  template <typename TrackerTraits, bool CountOnly = false>
  class GetDoubletsFromHisto {
  public:
    // #ifdef __CUDACC__
    //       __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)  // TODO: Alapakafy
    // #endif
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  uint32_t maxNumOfDoublets,
                                  CACell<TrackerTraits>* cells,
                                  uint32_t* nCells,
                                  caStructures::HitsMultiView hh,
                                  ::reco::CAGraphSoAConstView cc,
                                  ::reco::CALayersSoAConstView ll,
                                  ::reco::CADoubletCutsSoAConstView doubletCuts,
                                  uint32_t const* __restrict__ offsets,
                                  PhiBinner<TrackerTraits> const* phiBinner,
                                  HitToCell* outerHitHisto,
                                  uint32_t* __restrict__ pipelineCounters) const {
      doubletsFromHisto<TrackerTraits, CountOnly>(acc,
                                                  maxNumOfDoublets,
                                                  cells,
                                                  nCells,
                                                  hh,
                                                  cc,
                                                  ll,
                                                  doubletCuts,
                                                  offsets,
                                                  phiBinner,
                                                  outerHitHisto,
                                                  pipelineCounters);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoublets_h
