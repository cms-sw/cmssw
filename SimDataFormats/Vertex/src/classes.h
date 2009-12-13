#include "SimDataFormats/Vertex/interface/CoreSimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>
 
namespace {
  struct dictionary {
    SimVertex dummy33;
    std::vector<SimVertex> dummy333;
    edm::Wrapper<edm::SimVertexContainer> dummy33333;
    std::vector<const SimVertex*> dummyvcp;
    edm::SimVertexRef r1;
    edm::SimVertexRefVector rv1;
    edm::SimVertexRefProd rp1; 
  };
}
