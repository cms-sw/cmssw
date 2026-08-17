#ifndef RecoTracker_PixelSeeding_interface_alpaka_StackedModuleGeometrySoACollection_h
#define RecoTracker_PixelSeeding_interface_alpaka_StackedModuleGeometrySoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryDevice.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::StackedModuleGeometryDevice;
  using ::reco::StackedModuleGeometryHost;
  using StackedModuleGeometrySoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                                StackedModuleGeometryHost,
                                                                StackedModuleGeometryDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::StackedModuleGeometrySoACollection, reco::StackedModuleGeometryHost);

#endif  // RecoTracker_PixelSeeding_interface_alpaka_StackedModuleGeometrySoACollection_h
