#ifndef _VertexReconstructor_H_
#define _VertexReconstructor_H_

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include <vector>

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
  std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> &) const = 0; 

  virtual VertexReconstructor * clone() const = 0;

};

#endif
