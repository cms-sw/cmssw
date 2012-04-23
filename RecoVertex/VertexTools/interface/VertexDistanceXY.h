#ifndef Vertex_VertexDistanceXY_H
#define Vertex_VertexDistanceXY_H

#include "RecoVertex/VertexTools/interface/VertexDistance.h"

/**
 * Computes the distance between two vertices as the chi-squared formed 
 * with their positions in the transverse plane.
 */

class VertexDistanceXY : public VertexDistance {

public:

  VertexDistanceXY() : theNullMatrix(3, 0) {}

  /**
   * The signed distance is computed using a vector
   * from the primary to the secondary vertex and
   * a given reference vector.
   * The sign is determined by the scalar product of the x,y component of
   * the vector connecting the vertices and the reference vector:
   * if the scalar product is greater than zero, the sign is +1, else -1
   */
  virtual Measurement1D signedDistance(const reco::Vertex &primVtx , 
				 const reco::Vertex &secVtx,
				 const GlobalVector & momentum) const;

  virtual VertexDistanceXY * clone() const
  {
    return new VertexDistanceXY(*this);
  }


  using VertexDistance::distance;
  using VertexDistance::compatibility;

private:

  AlgebraicSymMatrix theNullMatrix;

  virtual Measurement1D distance(const GlobalPoint & vtx1Position, 
				 const GlobalError & vtx1PositionError, 
				 const GlobalPoint & vtx2Position, 
				 const GlobalError & vtx2PositionError) const;

  virtual float compatibility(const GlobalPoint & vtx1Position, 
			      const GlobalError & vtx1PositionError, 
			      const GlobalPoint & vtx2Position, 
			      const GlobalError & vtx2PositionError) const;
};


#endif



