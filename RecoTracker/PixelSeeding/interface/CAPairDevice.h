#ifndef RecoTracker_PixelSeeding_interface_CAPairDevice_H
#define RecoTracker_PixelSeeding_interface_CAPairDevice_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace caStructures {
  template <typename TDev>
  using CAPairDevice = PortableDeviceCollection<TDev, CAPairSoA>;
}

#endif  // RecoTracker_PixelSeeding_interface_CAPairDevice_H
