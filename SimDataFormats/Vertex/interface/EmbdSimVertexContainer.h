#ifndef EmbdSimVertexContainer_H
#define EmbdSimVertexContainer_H

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"

#include <vector>
#include <string>
 
namespace edm 
{
    class EmbdSimVertexContainer: public EDProduct 
    {
    public:
	typedef std::vector<EmbdSimVertex> SimVertexContainer;
	void insertVertex(EmbdSimVertex & v) { data.push_back(v); }
    private:
	SimVertexContainer data;
    };
} 
 

#endif
