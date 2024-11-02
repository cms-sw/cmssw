#include <alpaka/alpaka.hpp>

#include "RecoTracker/PixelSeeding/interface/CAParamsDevice.h"
#include "RecoTracker/PixelSeeding/interface/CAParamsHost.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAParamsSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testParamsSoA {

//   class TestFillKernel {
//   public:
//     template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
//     ALPAKA_FN_ACC void operator()(TAcc const& acc,
//                                   reco::CALayersSoAView layers_view,
//                                   reco::CACellsSoAView pairs_view,
//                                   reco::CARegionsSoAView regions_view) const {
//       if (cms::alpakatools::once_per_grid(acc)) {
//         CAParams_view.nvFinal() = 420;
//       }

//       for (int32_t j : cms::alpakatools::uniform_elements(acc, CAParams_view.metadata().size())) {
//         CAParams_view[j].zv() = (float)j;
//         CAParams_view[j].wv() = (float)j;
//         CAParams_view[j].chi2() = (float)j;
//         CAParams_view[j].ptv2() = (float)j;
//         CAParams_view[j].sortInd() = (uint16_t)j;
//       }
//       for (int32_t j : cms::alpakatools::uniform_elements(acc, ztracks_view.metadata().size())) {
//         ztracks_view[j].idv() = (int16_t)j;
//         ztracks_view[j].ndof() = (int32_t)j;
//       }
//     }
//   };

//   class TestVerifyKernel {
//   public:
//     template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
//     ALPAKA_FN_ACC void operator()(TAcc const& acc,
//                                   reco::CAParamsSoAView CAParams_view,
//                                   reco::CAParamsTracksSoAView ztracks_view) const {
//       if (cms::alpakatools::once_per_grid(acc)) {
//         ALPAKA_ASSERT_ACC(CAParams_view.nvFinal() == 420);
//       }

//       for (int32_t j : cms::alpakatools::uniform_elements(acc, CAParams_view.nvFinal())) {
//         ALPAKA_ASSERT(CAParams_view[j].zv() - (float)j < 0.0001);
//         ALPAKA_ASSERT(CAParams_view[j].wv() - (float)j < 0.0001);
//         ALPAKA_ASSERT(CAParams_view[j].chi2() - (float)j < 0.0001);
//         ALPAKA_ASSERT(CAParams_view[j].ptv2() - (float)j < 0.0001);
//         ALPAKA_ASSERT(CAParams_view[j].sortInd() == uint32_t(j));
//       }
//       for (int32_t j : cms::alpakatools::uniform_elements(acc, ztracks_view.metadata().size())) {
//         ALPAKA_ASSERT(ztracks_view[j].idv() == j);
//         ALPAKA_ASSERT(ztracks_view[j].ndof() == j);
//       }
//     }
//   };

  void runKernels(reco::CALayersSoAView layers_view,
                                  reco::CACellsSoAView pairs_view,
                                  Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(CAParams_view.metadata().size(), items);
    // auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    // alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, CAParams_view, ztracks_view);
    // alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, CAParams_view, ztracks_view);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testCAParamsSoAT