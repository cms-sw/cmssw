#ifndef _VertexReconstructor_H_
#define _VertexReconstructor_H_

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
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

  /** Reconstruct vertices, exploiting the beamspot constraint
   *  for the primary vertex.
   *  This trivial implementation is overwritable by the
   *  concrete reconstructors ...
   */
  virtual std::vector<TransientVertex>
    vertices( const std::vector<reco::TransientTrack> & t, const
              reco::BeamSpot & ) const
  {
     return vertices ( t );
  }

  virtual VertexReconstructor * clone() const = 0;

};

#endif
