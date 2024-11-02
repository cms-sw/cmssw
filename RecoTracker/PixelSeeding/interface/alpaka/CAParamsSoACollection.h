#ifndef RecoTracker_PixelSeeding_interface_CAParamsSoACollection_h
#define RecoTracker_PixelSeeding_interface_CAParamsSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAParamsDevice.h"
#include "RecoTracker/PixelSeeding/interface/CAParamsHost.h"
#include "RecoTracker/PixelSeeding/interface/CAParamsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

    using ::reco::CAParamsHost;
    using ::reco::CAParamsDevice;
    using CAParamsSoACollection =
        std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, CAParamsHost, CAParamsDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::CAParamsSoACollection, reco::CAParamsHost);

#endif  // RecoTracker_PixelSeeding_interface_CAParamsSoACollection_h