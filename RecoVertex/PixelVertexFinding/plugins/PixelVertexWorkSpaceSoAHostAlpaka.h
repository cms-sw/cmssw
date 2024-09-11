#ifndef RecoVertex_PixelVertexFinding_plugins_PixelVertexWorkSpaceSoAHostAlpaka_h
#define RecoVertex_PixelVertexFinding_plugins_PixelVertexWorkSpaceSoAHostAlpaka_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoVertex/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"

namespace vertexFinder {

  using PixelVertexWorkSpaceSoAHost = PortableHostCollection<PixelVertexWSSoALayout<>>;

}  // namespace vertexFinder

#endif  // RecoVertex_PixelVertexFinding_plugins_PixelVertexWorkSpaceSoAHostAlpaka_h
