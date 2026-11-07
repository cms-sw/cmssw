#ifndef RecoTracker_PixelSeeding_interface_StackedModuleGeometryHost_h
#define RecoTracker_PixelSeeding_interface_StackedModuleGeometryHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
  using StackedModuleGeometryHost = PortableHostCollection<StackedModuleGeometrySoA>;
}

#endif  // RecoTracker_PixelSeeding_interface_StackedModuleGeometryHost_h
