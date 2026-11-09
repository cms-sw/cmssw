#ifndef RecoTracker_PixelSeeding_interface_TripletDumpHost_h
#define RecoTracker_PixelSeeding_interface_TripletDumpHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/PixelSeeding/interface/TripletDumpSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace caStructures {
  using TripletDumpHost = PortableHostCollection<TripletDumpSoA>;
}  // namespace caStructures
#endif  // RecoTracker_PixelSeeding_interface_TripletDumpHost_h
