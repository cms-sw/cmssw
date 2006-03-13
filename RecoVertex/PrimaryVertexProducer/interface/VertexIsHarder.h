#ifndef VertexIsHarder_H
#define VertexIsHarder_H

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

/** \class VertexIsHarder
 * operator for sorting TransientVertex objects
 * in decreasing order of the sum of the squared track pT's
 */
struct VertexIsHarder {

  bool operator() ( const TransientVertex & v1, 
		    const TransientVertex & v2) const;


private:

  double sumPtSquared(const vector<TransientTrack> & tks) const;
};

#endif
