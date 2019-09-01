#include "RecoVertex/VertexPrimitives/interface/BasicVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

std::vector<VertexState> BasicVertexState::components() const {
  std::vector<VertexState> result;
  result.emplace_back(const_cast<BasicVertexState*>(this));
  return result;
}
