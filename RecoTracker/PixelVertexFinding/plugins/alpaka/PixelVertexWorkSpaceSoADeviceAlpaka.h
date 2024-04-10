#ifndef RecoTracker_PixelVertexFinding_plugins_alpaka_PixelVertexWorkSpaceSoADeviceAlpaka_h
#define RecoTracker_PixelVertexFinding_plugins_alpaka_PixelVertexWorkSpaceSoADeviceAlpaka_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "RecoTracker/PixelVertexFinding/plugins/PixelVertexWorkSpaceSoAHostAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace vertexFinder {

    using PixelVertexWorkSpaceSoADevice = PortableCollection<::vertexFinder::PixelVertexWSSoALayout<>>;
    using PixelVertexWorkSpaceSoAHost = ::vertexFinder::PixelVertexWorkSpaceSoAHost;

  }  // namespace vertexFinder

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelVertexFinding_plugins_alpaka_PixelVertexWorkSpaceSoADeviceAlpaka_h
