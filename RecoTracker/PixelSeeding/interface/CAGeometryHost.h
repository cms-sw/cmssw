#ifndef RecoTracker_PixelSeeding_interface_CAGeometryHost_H
#define RecoTracker_PixelSeeding_interface_CAGeometryHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
    using CAGeometryHost = PortableHostMultiCollection<CALayersSoA, CAGraphSoA, CAModulesSoA>;
}
#endif  // RecoTracker_PixelSeeding_interface_CAGeometryHost_H