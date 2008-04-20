#ifndef CoreSimVertex_H
#define CoreSimVertex_H

 
#include <CLHEP/Vector/LorentzVector.h>

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
 
#include <cmath>
 
/**  a generic Simulated Vertex
 */
class CoreSimVertex 
{ 
public:
    /// constructors
    CoreSimVertex() {}

    CoreSimVertex(const Hep3Vector & v, float tof) 
    // { theVertex[0] = v[0]; theVertex[1] = v[1]; theVertex[2] = v[2]; theVertex[3] = tof; }
    { theVertex.SetXYZT( v[0], v[1], v[2], tof ) ; }

    CoreSimVertex( const math::XYZVectorD & v, float tof )
    { theVertex.SetXYZT( v.x(), v.y(), v.z(), tof) ; }
    
    CoreSimVertex( const math::XYZTLorentzVectorD& v ) 
    { theVertex.SetXYZT( v.x(), v.y(), v.z(), v.t() ) ; }


    explicit CoreSimVertex(const HepLorentzVector & v) 
    //{ theVertex[0] = v.x(); theVertex[1] = v.y(); theVertex[2] = v.z(); theVertex[3] = v.t(); }
    { theVertex.SetXYZT( v.x(), v.y(), v.z(), v.t() ); }


    // bad trick, the compaler complains...
    // but seems to work so far (will re-check later)
    //
    HepLorentzVector position() { return HepLorentzVector(theVertex.x(),
                                                          theVertex.y(),
						          theVertex.z(),
							  theVertex.t()); }

     const math::XYZTLorentzVectorD& position() const { return theVertex; }
     // math::XYZTLorentzVectorD& position() { return theVertex; }
    

    void setEventId(EncodedEventId e) {eId=e;}
    EncodedEventId eventId() const {return eId;}

private:
    EncodedEventId eId;
    // HepLorentzVector theVertex;
     math::XYZTLorentzVectorD theVertex ;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const CoreSimVertex & v);

#endif 
