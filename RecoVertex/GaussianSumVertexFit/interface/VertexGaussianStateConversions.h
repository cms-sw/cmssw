#ifndef VertexGaussianStateConversions_H_
#define VertexGaussianStateConversions_H_
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"

namespace GaussianStateConversions {
  MultiGaussianState<3> multiGaussianStateFromVertex (const VertexState aState);
  VertexState vertexFromMultiGaussianState (const MultiGaussianState<3>& multiState);
}

#endif
