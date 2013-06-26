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
  virtual std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> &) const = 0; 

  /** Reconstruct vertices, exploiting the beamspot constraint
   *  for the primary vertex
   */
  virtual std::vector<TransientVertex> 
    vertices( const std::vector<reco::TransientTrack> & t, const 
              reco::BeamSpot & ) const
  {
    return vertices ( t );
  }

  /** Reconstruct vertices, but exploit the fact that you know
   *  that some tracks cannot come from a secondary vertex.
   *  \paramname primaries Tracks that _cannot_ come from
   *  a secondary vertex (but can, in principle, be
   *  non-primaries, also).
   *  \paramname tracks These are the tracks that are of unknown origin. These
   *  tracks are subjected to pattern recognition.
   *  \paramname spot A beamspot constraint is mandatory in this method.
   */
  virtual std::vector<TransientVertex>
    vertices( const std::vector<reco::TransientTrack> & primaries,
        const std::vector<reco::TransientTrack> & tracks,
        const reco::BeamSpot & spot ) const 
  {
    return vertices ( tracks, spot );
  }

  virtual VertexReconstructor * clone() const = 0;

};

#endif
