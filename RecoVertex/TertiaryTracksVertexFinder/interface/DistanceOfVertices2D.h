#ifndef DISTANCEOFVERTICES2D_H
#define DISTANCEOFVERTICES2D_H 

#include "RecoVertex/TertiaryTracksVertexFinder/interface/DistanceOfVertices.h"

// class to compute distance, error, significance of 2 Vertices
// in the XY plane

class DistanceOfVertices2D : public DistanceOfVertices {

public:

 DistanceOfVertices2D  () {}
  virtual ~DistanceOfVertices2D () {}

  virtual Measurement1D distance       ( const TransientVertex & , const TransientVertex &                )  ;
  virtual Measurement1D signedDistance ( const TransientVertex & , const TransientVertex & , const Hep3Vector & )  ;

 private:
  
};

#endif

