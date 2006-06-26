#ifndef SimVertexContainer_H
#define SimVertexContainer_H

#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

#include <vector>
 
namespace edm 
{
  typedef std::vector<SimVertex> SimVertexContainer;
  typedef edm::Ref<std::vector<SimVertex> > SimVertexRef ;
  typedef edm::RefProd<std::vector<SimVertex> > SimVertexRefProd;
  typedef edm::RefVector<std::vector<SimVertex> > SimVertexRefVector;

} 
 

#endif
