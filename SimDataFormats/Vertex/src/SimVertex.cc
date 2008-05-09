#include "SimDataFormats/Vertex/interface/SimVertex.h"

SimVertex::SimVertex() {}
 
SimVertex::SimVertex(const math::XYZVectorD& v, float tof) :
    Core(v,tof), itrack(-1) {}
 
SimVertex::SimVertex(const math::XYZVectorD& v, float tof, int it) :
    Core(v,tof), itrack(it) {}
 
SimVertex::SimVertex(const CoreSimVertex & v, int it) :
    Core(v), itrack(it) {}
 
std::ostream & operator <<(std::ostream & o , const SimVertex & v) 
{ return o << (SimVertex::Core)(v) << ", " <<  v.parentIndex(); }
