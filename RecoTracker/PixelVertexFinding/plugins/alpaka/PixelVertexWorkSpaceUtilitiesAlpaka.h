#ifndef RecoTracker_PixelVertexFinding_PixelVertexWorkSpaceUtilitiesAlpaka_h
#define RecoTracker_PixelVertexFinding_PixelVertexWorkSpaceUtilitiesAlpaka_h

#include <alpaka/alpaka.hpp>
#include "RecoTracker/PixelVertexFinding/plugins/PixelVertexWorkSpaceLayout.h"

// Methods that operate on View and ConstView of the PixelVertexWorkSpaceSoALayout.
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace vertexFinder {
    namespace workSpace {
      namespace utilities {
        using namespace ::vertexFinder::workSpace;

        ALPAKA_FN_ACC ALPAKA_FN_INLINE void init(PixelVertexWorkSpaceSoAView &workspace_view) {
          workspace_view.ntrks() = 0;
          workspace_view.nvIntermediate() = 0;
        }
      }  // namespace utilities
    }    // namespace workSpace
  }      // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
