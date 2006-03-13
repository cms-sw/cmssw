#ifndef _VertexReconstructor_H_
#define _VertexReconstructor_H_

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientTrack.h"

/** Abstract class for vertex reconstructors, 
 *  i.e. objects reconstructing vertices using a set of TransientTracks
 */

class VertexReconstructor {

public:

  VertexReconstructor() {}
  virtual ~VertexReconstructor() {}

  /** Reconstruct vertices
   */
  virtual 
  vector<TransientVertex> vertices(const vector<reco::TransientTrack> &) 
    const = 0; 

  virtual VertexReconstructor * clone() const = 0;

};

#endif
