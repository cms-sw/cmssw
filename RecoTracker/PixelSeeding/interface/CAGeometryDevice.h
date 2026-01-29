#ifndef RecoTracker_PixelSeeding_interface_CAGeometryDevice_H
#define RecoTracker_PixelSeeding_interface_CAGeometryDevice_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
  template <typename TDev>
  using CAGeometryDevice = PortableDeviceCollection<TDev, CALayout>;
}
#endif  // RecoTracker_PixelSeeding_interface_CAGeometryDevice_H
