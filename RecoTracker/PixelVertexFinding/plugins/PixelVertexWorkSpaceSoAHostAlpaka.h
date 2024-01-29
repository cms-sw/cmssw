#ifndef RecoTracker_PixelVertexFinding_plugins_PixelVertexWorkSpaceSoAHostAlpaka_h
#define RecoTracker_PixelVertexFinding_plugins_PixelVertexWorkSpaceSoAHostAlpaka_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"

namespace vertexFinder {

  using PixelVertexWorkSpaceSoAHost = PortableHostCollection<PixelVertexWSSoALayout<>>;

}  // namespace vertexFinder

#endif  // RecoTracker_PixelVertexFinding_plugins_PixelVertexWorkSpaceSoAHostAlpaka_h
