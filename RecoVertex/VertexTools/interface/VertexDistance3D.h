#ifndef Vertex_VertexDistance3D_H
#define Vertex_VertexDistance3D_H

#include "RecoVertex/VertexTools/interface/VertexDistance.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

/**
 * Computes the distance and chi-squared compatibility between two vertices
 * with their 3D positions.
 */

class VertexDistance3D : public VertexDistance {

public:
  using VertexDistance::compatibility;

  VertexDistance3D() : theNullMatrix(3, 0) {}

  /**
   * The signed distance is computed using a vector
   * from the primary to the secondary vertex and
   * a given reference vector.
   * The sign is determined by the scalar product  of
   * the vector connecting the vertices and the reference vector:
   * if the scalar product is greater than zero, the sign is +1, else -1
   */
  virtual Measurement1D signedDistance(const reco::Vertex &primVtx , 
				 const reco::Vertex &secVtx,
				 const GlobalVector & momentum) const;

  virtual VertexDistance3D * clone() const
  {
    return new VertexDistance3D(*this);
  }

  using VertexDistance::distance;

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



