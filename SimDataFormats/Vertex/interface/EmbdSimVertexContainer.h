#ifndef EmbdSimVertexContainer_H
#define EmbdSimVertexContainer_H

#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

#include <vector>
 
namespace edm 
{
  typedef std::vector<EmbdSimVertex> EmbdSimVertexContainer;
  typedef edm::Ref<std::vector<EmbdSimVertex> > EmbdSimVertexRef ;
  typedef edm::RefProd<std::vector<EmbdSimVertex> > EmbdSimVertexRefProd;
  typedef edm::RefVector<std::vector<EmbdSimVertex> > EmbdSimVertexRefVector;

} 
 

#endif
