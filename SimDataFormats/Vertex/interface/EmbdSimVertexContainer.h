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
	void insertVertex(EmbdSimVertex & v) { data.push_back(v); }
	void clear() { data.clear(); }
    private:
	SimVertexContainer data;
    };
} 
 

#endif
