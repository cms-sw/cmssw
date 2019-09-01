#ifndef RecoVertex_ConvertError_h
#define RecoVertex_ConvertError_h
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace RecoVertex {
  inline reco::Vertex::Error convertError(const GlobalError& ge) {
    reco::Vertex::Error error;
    error(0, 0) = ge.cxx();
    error(0, 1) = ge.cyx();
    error(0, 2) = ge.czx();
    error(1, 1) = ge.cyy();
    error(1, 2) = ge.czy();
    error(2, 2) = ge.czz();
    return error;
  }

  inline GlobalError convertError(const reco::Vertex::Error& error) {
    return GlobalError(error(0, 0), error(0, 1), error(1, 1), error(0, 2), error(1, 2), error(2, 2));
  }
}  // namespace RecoVertex

#endif
