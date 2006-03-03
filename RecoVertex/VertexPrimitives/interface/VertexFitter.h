#ifndef _VertexFitter_H_
#define _VertexFitter_H_

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include <vector>

/** 
 * Pure abstract base class for VertexFitters. 
 * Fits a CachingVertex using either:
 *  - TransientTracks; 
 *  - VertexTracks. 
 * A linearization point can be specified, 
 * or a prior estimate of the vertex position and error. 
 */

class VertexFitter {

public:

  VertexFitter() {}

  virtual ~VertexFitter() {}

  /** Fit vertex out of a set of TransientTracks
   */
  virtual CachingVertex 
  vertex(const vector<reco::TransientTrack> & tracks) const = 0;

  /** Fit vertex out of a set of VertexTracks
   */
  virtual CachingVertex 
  vertex(const vector<RefCountedVertexTrack> & tracks) const = 0;

  /** Fit vertex out of a set of TransientTracks. 
   *  Uses the specified linearization point.
   */
  virtual CachingVertex 
  vertex(const vector<reco::TransientTrack> & tracks, const GlobalPoint& linPoint) const = 0;

  /** Fit vertex out of a set of TransientTracks. 
   *  Uses the specified point as both the linearization point AND as prior
   *  estimate of the vertex position. The error is used for the 
   *  weight of the prior estimate.
   */
  virtual CachingVertex 
  vertex(const vector<reco::TransientTrack> & tracks, const GlobalPoint& priorPos,
  	 const GlobalError& priorError) const = 0;

  /** Fit vertex out of a set of VertexTracks.
   *  Uses the specified point and error as the prior estimate of the vertex.
   *  This position is not used to relinearize the tracks.
   */
  virtual CachingVertex 
  vertex(const vector<RefCountedVertexTrack> & tracks, 
	 const GlobalPoint& priorPos,
	 const GlobalError& priorError) const = 0;

  /** Fit vertex out of a VertexSeed
   */
//   virtual CachingVertex 
//   vertex(const RefCountedVertexSeed vtxSeed) const = 0;

  virtual VertexFitter * clone() const = 0;

};

#endif
