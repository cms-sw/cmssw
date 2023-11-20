#ifndef RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoAHost_h
#define RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoAHost_h
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexUtilities.h"
#include "PixelVertexWorkSpaceLayout.h"

template <int32_t S>
class PixelVertexWorkSpaceSoAHost : public PortableHostCollection<PixelVertexWSSoALayout<>> {
public:
  //explicit PixelVertexWorkSpaceSoAHost() : PortableHostCollection<PixelVertexWSSoALayout<>>(S) {}

  // Constructor which specifies the SoA size and CUDA stream
  template <typename TQueue>
  explicit PixelVertexWorkSpaceSoAHost(TQueue queue) : PortableHostCollection<PixelVertexWSSoALayout<>>(S, queue) {}

  explicit PixelVertexWorkSpaceSoAHost(alpaka_common::DevHost const& host)
      : PortableHostCollection<PixelVertexWSSoALayout<>>(S, host) {}
};

namespace vertexFinder {
  namespace workSpace {
    using PixelVertexWorkSpaceSoAHost = PixelVertexWorkSpaceSoAHost<zVertex::MAXTRACKS>;
  }
}  // namespace vertexFinder
#endif
