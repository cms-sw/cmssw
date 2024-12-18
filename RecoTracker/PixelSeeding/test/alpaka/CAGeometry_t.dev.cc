#include <alpaka/alpaka.hpp>

#include "RecoTracker/PixelSeeding/interface/CAGeometryDevice.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testParamsSoA {

//   class TestFillKernel {
//   public:
//     template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
//     ALPAKA_FN_ACC void operator()(TAcc const& acc,
//                                   reco::CALayersSoAView layers_view,
//                                   reco::CAGraphSoAView pairs_view,
//                                   reco::CARegionsSoAView regions_view) const {
//       if (cms::alpakatools::once_per_grid(acc)) {
//         CAGeometry_view.nvFinal() = 420;
//       }

//       for (int32_t j : cms::alpakatools::uniform_elements(acc, CAGeometry_view.metadata().size())) {
//         CAGeometry_view[j].zv() = (float)j;
//         CAGeometry_view[j].wv() = (float)j;
//         CAGeometry_view[j].chi2() = (float)j;
//         CAGeometry_view[j].ptv2() = (float)j;
//         CAGeometry_view[j].sortInd() = (uint16_t)j;
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
//                                   reco::CAGeometrySoAView CAGeometry_view,
//                                   reco::CAGeometryTracksSoAView ztracks_view) const {
//       if (cms::alpakatools::once_per_grid(acc)) {
//         ALPAKA_ASSERT_ACC(CAGeometry_view.nvFinal() == 420);
//       }

//       for (int32_t j : cms::alpakatools::uniform_elements(acc, CAGeometry_view.nvFinal())) {
//         ALPAKA_ASSERT(CAGeometry_view[j].zv() - (float)j < 0.0001);
//         ALPAKA_ASSERT(CAGeometry_view[j].wv() - (float)j < 0.0001);
//         ALPAKA_ASSERT(CAGeometry_view[j].chi2() - (float)j < 0.0001);
//         ALPAKA_ASSERT(CAGeometry_view[j].ptv2() - (float)j < 0.0001);
//         ALPAKA_ASSERT(CAGeometry_view[j].sortInd() == uint32_t(j));
//       }
//       for (int32_t j : cms::alpakatools::uniform_elements(acc, ztracks_view.metadata().size())) {
//         ALPAKA_ASSERT(ztracks_view[j].idv() == j);
//         ALPAKA_ASSERT(ztracks_view[j].ndof() == j);
//       }
//     }
//   };

  void runKernels(reco::CALayersSoAView layers_view,
                                  reco::CAGraphSoAView pairs_view,
                                  Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(CAGeometry_view.metadata().size(), items);
    // auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    // alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, CAGeometry_view, ztracks_view);
    // alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, CAGeometry_view, ztracks_view);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testCAGeometrySoAT