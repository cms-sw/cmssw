#ifndef CoreSimVertex_H
#define CoreSimVertex_H
 
#include <CLHEP/Vector/LorentzVector.h>
 
#include <cmath>
 
/**  a generic Simulated Vertex
 */
class CoreSimVertex 
{ 
public:
    /// constructors
    CoreSimVertex() {}
    CoreSimVertex(const Hep3Vector & v, float tof) 
    { theVertex[0] = v[0]; theVertex[1] = v[1]; theVertex[2] = v[2]; theVertex[3] = tof; }
    explicit CoreSimVertex(const HepLorentzVector & v) 
    { theVertex[0] = v.x(); theVertex[1] = v.y(); theVertex[2] = v.z(); theVertex[3] = v.t(); }
    HepLorentzVector position() const { return theVertex; }
private:
    HepLorentzVector theVertex;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const CoreSimVertex & v);

#endif 
