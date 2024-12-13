#ifndef RecoTracker_PixelSeeding_interface_CAGeometrySoACollection_h
#define RecoTracker_PixelSeeding_interface_CAGeometrySoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryDevice.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

    using ::reco::CAGeometryHost;
    using ::reco::CAGeometryDevice;
    using CAGeometrySoACollection =
        std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, CAGeometryHost, CAGeometryDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::CAGeometrySoACollection, reco::CAGeometryHost);

#endif  // RecoTracker_PixelSeeding_interface_CAGeometrySoACollection_h