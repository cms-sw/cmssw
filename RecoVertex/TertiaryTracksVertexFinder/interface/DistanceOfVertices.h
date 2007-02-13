#ifndef DISTANCEOFVERTICES_H  
#define DISTANCEOFVERTICES_H 

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"
#include "CLHEP/Vector/ThreeVector.h"

// class to compute distance, error, significance of 2 Vertices
// This is the ABC

class DistanceOfVertices {

public:

  DistanceOfVertices  () {}
  virtual ~DistanceOfVertices () {}

  virtual Measurement1D distance       ( const TransientVertex & , const TransientVertex &                      ) = 0 ;
  virtual Measurement1D signedDistance ( const TransientVertex & , const TransientVertex & , const Hep3Vector & ) = 0 ;
  
  
protected:

  Hep3Vector   deltaV2V1 ( const TransientVertex & v1 , const TransientVertex & v2 ) {
    return Hep3Vector ( v2.position().x() - v1.position().x() ,
			v2.position().y() - v1.position().y() ,
			v2.position().z() - v1.position().z() ) ;
  }

  
  double terms2D ( const TransientVertex & v1 , const TransientVertex & v2 ) {
    // better to have meaningful variables ...
    double deltaX = deltaV2V1(v1,v2).x() ;
    double deltaY = deltaV2V1(v1,v2).y() ;

    // covariance matrix of the diff. = sum of ind. cov. matr.
    GlobalError covDelta = v1.positionError() + v2.positionError() ;

    double covDeltaXX = covDelta.cxx() ;
    double covDeltaYY = covDelta.cyy() ;
    double covDeltaXY = covDelta.cyx() ;

    return (   deltaX*deltaX * covDeltaXX  +
	       deltaY*deltaY * covDeltaYY  +
	     2*deltaX*deltaY * covDeltaXY    ) ; 
  }


  double terms3D ( const TransientVertex & v1 , const TransientVertex & v2 ) {
    //
    double deltaX = deltaV2V1(v1,v2).x() ;
    double deltaY = deltaV2V1(v1,v2).y() ;
    double deltaZ = deltaV2V1(v1,v2).z() ;

    // covariance matrix of the diff. = sum of ind. cov. matr.
    GlobalError covDelta = v1.positionError() + v2.positionError() ;

    double covDeltaZZ = covDelta.czz() ;
    double covDeltaXZ = covDelta.czx() ;
    double covDeltaYZ = covDelta.czy() ;

    
    return (   terms2D ( v1 , v2 )         +
	       deltaZ*deltaZ * covDeltaZZ  +
	     2*deltaX*deltaZ * covDeltaXZ  +  
	     2*deltaY*deltaZ * covDeltaYZ   ) ;
  }
  
};
#endif

