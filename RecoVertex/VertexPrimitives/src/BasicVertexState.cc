#include "RecoVertex/VertexPrimitives/interface/BasicVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

std::vector<VertexState> 
BasicVertexState::components() const {
  std::vector<VertexState> result; result.reserve(1);
  result.push_back( const_cast<BasicVertexState*>(this));
  return result;
}

