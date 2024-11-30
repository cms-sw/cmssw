#ifndef RecoTracker_PixelSeeding_interface_CACoupleDevice_H
#define RecoTracker_PixelSeeding_interface_CACoupleDevice_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoTracker/PixelSeeding/interface/CACoupleSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace caStructures
{
    template <typename TDev>
    using CACoupleDevice = PortableDeviceCollection<CACoupleSoA, TDev>;
}

#endif  // RecoTracker_PixelSeeding_interface_CACoupleDevice_H