#ifndef _VertexFitter_H_
#define _VertexFitter_H_

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <vector>

/** 
 * Pure abstract base class for VertexFitters. 
 * Fits a CachingVertex using either:
 *  - TransientTracks; 
 *  - VertexTracks. 
 * A linearization point can be specified, 
 * or a prior estimate of the vertex position and error. 
 */

template <unsigned int N>
class VertexFitter {
public:
  VertexFitter() {}

  virtual ~VertexFitter() {}

  /** Fit vertex out of a set of TransientTracks
   */
  virtual CachingVertex<N> vertex(const std::vector<reco::TransientTrack>& tracks) const = 0;

  /** Fit vertex out of a set of VertexTracks. For the first iteration, the already 
   * linearized track will be used.
   */
  virtual CachingVertex<N> vertex(const std::vector<typename CachingVertex<N>::RefCountedVertexTrack>& tracks) const = 0;

  /** Same as above, only now also the
   * BeamSpot constraint is provided.
   */
  virtual CachingVertex<N> vertex(const std::vector<typename CachingVertex<N>::RefCountedVertexTrack>& tracks,
                                  const reco::BeamSpot& spot) const = 0;

  /** Fit vertex out of a set of TransientTracks. 
   *  The specified point will be used as linearization point, but will NOT be used as prior.
   */
  virtual CachingVertex<N> vertex(const std::vector<reco::TransientTrack>& tracks,
                                  const GlobalPoint& linPoint) const = 0;

  /** Fit vertex out of a set of TransientTracks. 
   *  Uses the specified point as both the linearization point AND as prior
   *  estimate of the vertex position. The error is used for the 
   *  weight of the prior estimate.
   */
  virtual CachingVertex<N> vertex(const std::vector<reco::TransientTrack>& tracks,
                                  const GlobalPoint& priorPos,
                                  const GlobalError& priorError) const = 0;

  /** Fit vertex out of a set of TransientTracks. 
   *  The specified BeamSpot will be used as priot, but NOT for the linearization.
   * The specified LinearizationPointFinder will be used to find the linearization point.
   */
  virtual CachingVertex<N> vertex(const std::vector<reco::TransientTrack>& tracks,
                                  const reco::BeamSpot& beamSpot) const = 0;

  /** Fit vertex out of a set of VertexTracks.
   *  Uses the specified point and error as the prior estimate of the vertex.
   *  This position is NOT used to relinearize the tracks.
   */
  virtual CachingVertex<N> vertex(const std::vector<typename CachingVertex<N>::RefCountedVertexTrack>& tracks,
                                  const GlobalPoint& priorPos,
                                  const GlobalError& priorError) const = 0;

  /** Fit vertex out of a VertexSeed
   */
  //   virtual CachingVertex<N>
  //   vertex(const RefCountedVertexSeed vtxSeed) const = 0;

  virtual VertexFitter* clone() const = 0;
};

#endif
