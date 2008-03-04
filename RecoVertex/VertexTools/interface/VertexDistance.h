#ifndef Tracker_VertexDistance_H
#define Tracker_VertexDistance_H

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/** \class VertexDistance
 *  Abstact class which defines a distance and compatibility between vertices.
 */

class VertexState;

class VertexDistance {
 public:

  virtual ~VertexDistance() {}

  virtual Measurement1D distance(const reco::Vertex &, 
				 const reco::Vertex &) const = 0;

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
			       const reco::Vertex &) const = 0;

  virtual Measurement1D distance(const VertexState &, 
				 const VertexState &) const = 0;

  virtual float compatibility (const VertexState &, 
			       const VertexState &) const = 0;

  virtual VertexDistance * clone() const = 0;

};
#endif  //  Tracker_VertexDistance_H
