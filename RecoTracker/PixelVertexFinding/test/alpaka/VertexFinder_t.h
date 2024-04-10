#ifndef RecoTracker_PixelVertexFinding_test_alpaka_VertexFinder_t_h
#define RecoTracker_PixelVertexFinding_test_alpaka_VertexFinder_t_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace vertexfinder_t {
    void runKernels(Queue& queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelVertexFinding_test_alpaka_VertexFinder_t_h
