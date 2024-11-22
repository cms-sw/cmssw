#ifndef RecoTracker_PixelSeeding_interface_CACoupleHost_h
#define RecoTracker_PixelSeeding_interface_CACoupleHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/PixelSeeding/interface/CACoupleSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// maybe not realy needed ...?
namespace caStructures {
    using CACoupleHost = PortableHostCollection<CACoupleSoA>;
}
#endif  // RecoTracker_PixelSeeding_interface_CACoupleHost_h