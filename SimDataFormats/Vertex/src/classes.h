#include "SimDataFormats/Vertex/interface/CoreSimVertex.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <CLHEP/Vector/LorentzVector.h>

#include <vector>
 
namespace {
namespace {
    CLHEP::HepLorentzVector dummy1;
    CLHEP::Hep3Vector dummy2;
    EmbdSimVertex dummy33;
    std::vector<EmbdSimVertex> dummy333;
    edm::Wrapper<edm::EmbdSimVertexContainer> dummy33333;
    EmbdSimVertex::EmbdSimVertexRef r1;
    EmbdSimVertex::EmbdSimVertexRefVector rv1;
    EmbdSimVertex::EmbdSimVertexRefProd rp1; 
}
}
