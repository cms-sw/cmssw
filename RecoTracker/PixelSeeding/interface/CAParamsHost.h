#ifndef RecoTracker_PixelSeeding_interface_CAParamsHost_H
#define RecoTracker_PixelSeeding_interface_CAParamsHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAParamsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
    using CAParamsHost = PortableHostMultiCollection<CALayersSoA, CACellsSoA, CARegionsSoA>;
}
#endif  // DataFormats_VertexSoA_CAParamsHost_H