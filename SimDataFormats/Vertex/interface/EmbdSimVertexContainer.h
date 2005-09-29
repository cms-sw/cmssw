#ifndef EmbdSimVertexContainer_H
#define EmbdSimVertexContainer_H

#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"

#include <vector>
#include <string>
 
namespace edm 
{
  class EmbdSimVertexContainer 
    {
    public:
      typedef std::vector<EmbdSimVertex> SimVertexContainer;
      void insertVertex(const EmbdSimVertex & v) { data.push_back(v); }
      void clear() { data.clear(); }
      unsigned int size () const {return data.size();}
      EmbdSimVertex operator[] (int i) const {return data[i]; }
 
      SimVertexContainer::const_iterator begin () const {return data.begin();}
      SimVertexContainer::const_iterator end () const  {return data.end();}

    private:
      SimVertexContainer data;
    };
} 
 

#endif
