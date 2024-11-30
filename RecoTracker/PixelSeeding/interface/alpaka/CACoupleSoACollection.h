#ifndef RecoTracker_PixelSeeding_interface_CACoupleSoACollection_h
#define RecoTracker_PixelSeeding_interface_CACoupleSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/PixelSeeding/interface/CACoupleDevice.h"
#include "RecoTracker/PixelSeeding/interface/CACoupleHost.h"
#include "RecoTracker/PixelSeeding/interface/CACoupleSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    using ::caStructures::CACoupleHost;
    using ::caStructures::CACoupleDevice;
    using CACoupleSoACollection =
        std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, CACoupleHost, CACoupleDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(CACoupleSoACollection, ::caStructures::CACoupleHost);

#endif  // RecoTracker_PixelSeeding_interface_CACoupleSoACollection_h