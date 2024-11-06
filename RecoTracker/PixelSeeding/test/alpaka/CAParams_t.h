#ifndef RecoTracker_PixelSeeding_test_alpaka_CAParamsSoA_test_h
#define RecoTracker_PixelSeeding_test_alpaka_CAParamsSoA_test_h

#include "RecoTracker/PixelSeeding/interface/CAParamsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testParamsSoA {

  void runKernels(::reco::CALayersSoAView layers_view,
                                  ::reco::CACellsSoAView pairs_view,
                                  Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testCAParamsSoA

#endif  // RecoTracker_PixelSeeding_test_alpaka_CAParamsSoA_test_h