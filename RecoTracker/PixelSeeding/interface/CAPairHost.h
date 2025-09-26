#ifndef RecoTracker_PixelSeeding_interface_CAPairHost_h
#define RecoTracker_PixelSeeding_interface_CAPairHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace caStructures {
  using CAPairHost = PortableHostCollection<CAPairSoA>;
}
#endif  // RecoTracker_PixelSeeding_interface_CAPairHost_h
