#ifndef RecoTracker_PixelSeeding_interface_TripletDumpDevice_H
#define RecoTracker_PixelSeeding_interface_TripletDumpDevice_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoTracker/PixelSeeding/interface/TripletDumpSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace caStructures {
  template <typename TDev>
  using TripletDumpDevice = PortableDeviceCollection<TDev, TripletDumpSoA>;
}  // namespace caStructures

#endif  // RecoTracker_PixelSeeding_interface_TripletDumpDevice_H
