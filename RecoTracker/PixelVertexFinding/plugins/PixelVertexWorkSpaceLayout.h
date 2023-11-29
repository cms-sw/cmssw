#ifndef RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceLayout_h
#define RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Intermediate data used in the vertex reco algos
// For internal use only
GENERATE_SOA_LAYOUT(PixelVertexWSSoALayout,
                    SOA_COLUMN(uint16_t, itrk),            // index of original track
                    SOA_COLUMN(float, zt),                 // input track z at bs
                    SOA_COLUMN(float, ezt2),               // input error^2 on the above
                    SOA_COLUMN(float, ptt2),               // input pt^2 on the above
                    SOA_COLUMN(uint8_t, izt),              // interized z-position of input tracks
                    SOA_COLUMN(int32_t, iv),               // vertex index for each associated track
                    SOA_SCALAR(uint32_t, ntrks),           // number of "selected tracks"
                    SOA_SCALAR(uint32_t, nvIntermediate))  // the number of vertices after splitting pruning etc.

namespace vertexFinder {
  namespace workSpace {
    using PixelVertexWorkSpaceSoALayout = PixelVertexWSSoALayout<>;
    using PixelVertexWorkSpaceSoAView = PixelVertexWSSoALayout<>::View;
    using PixelVertexWorkSpaceSoAConstView = PixelVertexWSSoALayout<>::ConstView;
  }  // namespace workSpace
}  // namespace vertexFinder

#endif
