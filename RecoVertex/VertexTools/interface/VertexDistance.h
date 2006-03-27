#ifndef Tracker_VertexDistance_H
#define Tracker_VertexDistance_H

#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

/** \class VertexDistance
 *  Abstact class which defines a distance and compatibility between vertices.
 */

class VertexState;

class VertexDistance {
 public:

  virtual ~VertexDistance() {}

  virtual Measurement1D distance(const reco::Vertex &, 
				 const reco::Vertex &) const = 0;

  virtual float compatibility (const reco::Vertex &, 
			       const reco::Vertex &) const = 0;

  virtual Measurement1D distance(const VertexState &, 
				 const VertexState &) const = 0;

  virtual float compatibility (const VertexState &, 
			       const VertexState &) const = 0;

  virtual VertexDistance * clone() const = 0;

};
#endif  //  Tracker_VertexDistance_H
