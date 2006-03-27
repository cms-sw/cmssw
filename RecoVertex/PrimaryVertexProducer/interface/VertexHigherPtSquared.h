#ifndef VertexHigherPtSquared_H
#define VertexHigherPtSquared_H

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

/** \class VertexHigherPtSquared
 * operator for sorting TransientVertex objects
 * in decreasing order of the sum of the squared track pT's
 */
struct VertexHigherPtSquared {

  bool operator() ( const TransientVertex & v1, 
		    const TransientVertex & v2) const;


private:

  double sumPtSquared(const vector<TransientTrack> & tks) const;
};

#endif
