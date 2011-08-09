#ifndef VertexHigherPtSquared_H
#define VertexHigherPtSquared_H

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <vector>

/** \class VertexHigherPtSquared
 * operator for sorting TransientVertex objects
 * in decreasing order of the sum of the squared track pT's
 */
struct VertexHigherPtSquared {

  bool operator() ( const TransientVertex & v1, 
		    const TransientVertex & v2) const;

  bool operator() ( const reco::Vertex & v1, const reco::Vertex & v2) const;


public:

  double sumPtSquared(const reco::Vertex & v) const;


};

#endif
