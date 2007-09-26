#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"

EmbdSimVertex::EmbdSimVertex() {}
 
EmbdSimVertex::EmbdSimVertex(const Hep3Vector & v, float tof) :
    Core(v,tof), itrack(-1) {}

EmbdSimVertex::EmbdSimVertex(const math::XYZVectorD& v, float tof) :
    Core(v,tof), itrack(-1) {}
 
EmbdSimVertex::EmbdSimVertex(const Hep3Vector & v, float tof, int it) :
    Core(v,tof), itrack(it) {}

EmbdSimVertex::EmbdSimVertex(const math::XYZVectorD& v, float tof, int it) :
    Core(v,tof), itrack(it) {}

 
EmbdSimVertex::EmbdSimVertex(const CoreSimVertex & v, int it) :
    Core(v), itrack(it) {}
 
std::ostream & operator <<(std::ostream & o , const EmbdSimVertex & v) 
{ return o << (EmbdSimVertex::Core)(v) << ", " <<  v.parentIndex(); }
