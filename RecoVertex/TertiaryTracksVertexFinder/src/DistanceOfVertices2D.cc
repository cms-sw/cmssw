
#include "RecoVertex/TertiaryTracksVertexFinder/interface/DistanceOfVertices2D.h"

Measurement1D DistanceOfVertices2D::distance ( const TransientVertex & v1 , const TransientVertex & v2 ) {
  
  double FlightDistance2D = deltaV2V1(v1,v2).perp() ; 
  
  // for the significances, we need the error on the difference:
  // => take into account the full correlation matrices of both vertices
  //    (but neglect correl. between them, they are not easily available)
  
  double sigmaFlightDistance2D2;
  if(  FlightDistance2D > 0.000001 )
    sigmaFlightDistance2D2 = ( 1 / ( FlightDistance2D*FlightDistance2D ) ) * terms2D ( v1 , v2 ) ; 
  else
    sigmaFlightDistance2D2 = 0.0;
  
  double sigmaFlightDistance2D ;
  if ( sigmaFlightDistance2D2 >= 0 ) {
    sigmaFlightDistance2D = sqrt ( sigmaFlightDistance2D2 ) ;
  }
  else {
    std::cout << "DistanceOfVertices2D::distance : sigmaFlightDistance2D2 <= 0 : " << sigmaFlightDistance2D2 << std::endl ;
    sigmaFlightDistance2D = 0.0 ;
  }

  return Measurement1D ( FlightDistance2D , sigmaFlightDistance2D ) ;
}


Measurement1D DistanceOfVertices2D::signedDistance ( const TransientVertex & v1 , const TransientVertex & v2 , const Hep3Vector & direction ) {
  // give a sign to the distance of Vertices v1 and v2:
  // + if (v2-v1) is along direction, - if in opposite direction
  
  Hep3Vector v1ToV2 = deltaV2V1 ( v1, v2 ) ;
  int sign2d = -1 ;
  if ( ( direction.x()*v1ToV2.x() + direction.y()*v1ToV2.y() ) > 0 )  sign2d = 1 ;
  
  Measurement1D unsignedFlightDistance = distance ( v1 , v2 ) ;

  return Measurement1D ( sign2d * unsignedFlightDistance.value() , unsignedFlightDistance.error() ) ; 
}

