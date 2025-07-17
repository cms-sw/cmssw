#ifndef RecoTracker_PixelSeeding_interface_CAPairSoACollection_h
#define RecoTracker_PixelSeeding_interface_CAPairSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAPairDevice.h"
#include "RecoTracker/PixelSeeding/interface/CAPairHost.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::caStructures::CAPairDevice;
  using ::caStructures::CAPairHost;
  using CAPairSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, CAPairHost, CAPairDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(CAPairSoACollection, ::caStructures::CAPairHost);

#endif  // RecoTracker_PixelSeeding_interface_CAPairSoACollection_h
