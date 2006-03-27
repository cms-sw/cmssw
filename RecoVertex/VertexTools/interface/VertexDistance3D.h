#ifndef Vertex_VertexDistance3D_H
#define Vertex_VertexDistance3D_H

#include "RecoVertex/VertexTools/interface/VertexDistance.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

/**
 * Computes the distance and chi-squared compatibility between two vertices
 * with their 3D positions.
 */

class VertexDistance3D : public VertexDistance {

public:
  VertexDistance3D() : theNullMatrix(3, 0) {}

  virtual Measurement1D distance(const reco::Vertex &, 
				 const reco::Vertex &) const;

  virtual float compatibility (const reco::Vertex &, 
			       const reco::Vertex &) const;

  virtual Measurement1D distance(const VertexState &, const VertexState &) const;

  virtual float compatibility (const VertexState &, const VertexState &) const;

  virtual VertexDistance3D * clone() const
  {
    return new VertexDistance3D(*this);
  }


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



