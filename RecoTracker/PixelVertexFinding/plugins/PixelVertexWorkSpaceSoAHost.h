#ifndef RecoTracker_PixelVertexFinding_plugins_PixelVertexWorkSpaceSoAHost_h
#define RecoTracker_PixelVertexFinding_plugins_PixelVertexWorkSpaceSoAHost_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "RecoTracker/PixelVertexFinding/plugins/PixelVertexWorkSpaceUtilities.h"

template <int32_t S>
class PixelVertexWorkSpaceSoAHost : public cms::cuda::PortableHostCollection<PixelVertexWSSoALayout<>> {
public:
  explicit PixelVertexWorkSpaceSoAHost() : PortableHostCollection<PixelVertexWSSoALayout<>>(S) {}
  // Constructor which specifies the SoA size and CUDA stream
  explicit PixelVertexWorkSpaceSoAHost(cudaStream_t stream)
      : PortableHostCollection<PixelVertexWSSoALayout<>>(S, stream) {}
};

namespace gpuVertexFinder {
  namespace workSpace {
    using PixelVertexWorkSpaceSoAHost = PixelVertexWorkSpaceSoAHost<zVertex::utilities::MAXTRACKS>;
  }
}  // namespace gpuVertexFinder
#endif
