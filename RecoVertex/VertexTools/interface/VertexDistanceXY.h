#ifndef Vertex_VertexDistanceXY_H
#define Vertex_VertexDistanceXY_H

#include "RecoVertex/VertexTools/interface/VertexDistance.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

/**
 * Computes the distance between two vertices as the chi-squared formed 
 * with their positions in the transverse plane.
 */

class VertexDistanceXY : public VertexDistance {

public:

  VertexDistanceXY() : theNullMatrix(3, 0) {}

  virtual Measurement1D distance(const reco::Vertex &, 
				 const reco::Vertex &) const;

  virtual float compatibility (const reco::Vertex &, 
			       const reco::Vertex &) const;

  virtual Measurement1D distance(const VertexState &, const VertexState &) const;

  virtual float compatibility (const VertexState &, const VertexState &) const;

  virtual VertexDistanceXY * clone() const
  {
    return new VertexDistanceXY(*this);
  }


protected:

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



