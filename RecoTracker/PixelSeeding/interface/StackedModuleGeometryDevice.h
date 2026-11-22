#ifndef RecoTracker_PixelSeeding_interface_StackedModuleGeometryDevice_h
#define RecoTracker_PixelSeeding_interface_StackedModuleGeometryDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
  template <typename TDev>
  using StackedModuleGeometryDevice = PortableDeviceCollection<TDev, StackedModuleGeometrySoA>;
}

#endif  // RecoTracker_PixelSeeding_interface_StackedModuleGeometryDevice_h
