#ifndef RecoTracker_PixelSeeding_interface_TripletDumpSoACollection_h
#define RecoTracker_PixelSeeding_interface_TripletDumpSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/PixelSeeding/interface/TripletDumpDevice.h"
#include "RecoTracker/PixelSeeding/interface/TripletDumpHost.h"
#include "RecoTracker/PixelSeeding/interface/TripletDumpSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::caStructures::TripletDumpDevice;
  using ::caStructures::TripletDumpHost;
  using TripletDumpSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, TripletDumpHost, TripletDumpDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(TripletDumpSoACollection, ::caStructures::TripletDumpHost);

#endif  // RecoTracker_PixelSeeding_interface_TripletDumpSoACollection_h
