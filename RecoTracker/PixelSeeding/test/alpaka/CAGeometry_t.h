#ifndef RecoTracker_PixelSeeding_test_alpaka_CAGeometrySoA_test_h
#define RecoTracker_PixelSeeding_test_alpaka_CAGeometrySoA_test_h

#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testParamsSoA {

  void runKernels(::reco::CALayersSoAView layers_view,
                                  ::reco::CAGraphSoAView pairs_view,
                                  Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testCAGeometrySoA

#endif  // RecoTracker_PixelSeeding_test_alpaka_CAGeometrySoA_test_h