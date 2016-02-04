#ifndef VertexDistance_H
#define VertexDistance_H

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

/** \class VertexDistance
 *  Abstact class which defines a distance and compatibility between vertices.
 */

class VertexState;

class VertexDistance {
 public:

  virtual ~VertexDistance() {}

  Measurement1D distance(const reco::Vertex &, 
			 const reco::Vertex &) const;

  Measurement1D distance(const VertexState &, 
			 const VertexState &) const;

  Measurement1D distance(const reco::Vertex &, 
			 const VertexState &) const;

  Measurement1D distance(const VertexState &, 
			 const reco::Vertex &) const;

  /**
   * The signed distance is computed using a vector
   * from the primary to the secondary vertex and
   * a given reference vector.
   * for the 2D case:
   *   The sign is determined by the scalar product of the x,y component of
   *   the vector connecting the vertices and the reference vector:
   *   if the scalar product is greater than zero, the sign is +1, else -1
   *
   * for the 3D case:
   *   Follows same approach, using all three components of the two vectors
   */
  virtual Measurement1D signedDistance(const reco::Vertex &primVtx , 
				       const reco::Vertex &secVtx,
				       const GlobalVector & momentum) const = 0;

  virtual float compatibility (const reco::Vertex &, 
			       const reco::Vertex &) const;

  virtual float compatibility (const VertexState &, 
			       const VertexState &) const;

  virtual float compatibility(const reco::Vertex &, 
			 const VertexState &) const;

  virtual float compatibility(const VertexState &, 
			 const reco::Vertex &) const;

  virtual VertexDistance * clone() const = 0;

protected: 
  virtual Measurement1D distance(const GlobalPoint & vtx1Position, 
				 const GlobalError & vtx1PositionError, 
				 const GlobalPoint & vtx2Position, 
				 const GlobalError & vtx2PositionError) const = 0;

  virtual float compatibility(const GlobalPoint & vtx1Position, 
			      const GlobalError & vtx1PositionError, 
			      const GlobalPoint & vtx2Position, 
			      const GlobalError & vtx2PositionError) const = 0;

};
#endif  //  Tracker_VertexDistance_H
