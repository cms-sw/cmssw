#ifndef _VertexTrackCompatibilityEstimator_H
#define _VertexTrackCompatibilityEstimator_H

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

/**
 * Pure abstract base class for compatibility estimators 
 * (distance, chi-squared, etc.)
 */

template <unsigned int N>
class VertexTrackCompatibilityEstimator {
 
public:

  typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef typename VertexTrack<N>::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;

  VertexTrackCompatibilityEstimator(){}
  virtual ~VertexTrackCompatibilityEstimator(){}
  
  /**
   * Methods giving back the compatibility estimation
   */
  virtual float estimate(const CachingVertex<N> & v, 
			 const RefCountedLinearizedTrackState track) const = 0;

  virtual float estimate(const reco::Vertex & v, 
			 const reco::TransientTrack & track) const = 0;

  // obsolete ?
  virtual float estimate(const CachingVertex<N> & v, 
			 const RefCountedVertexTrack track) const = 0;
  /**
   * Clone method 
   */
  virtual VertexTrackCompatibilityEstimator<N> * clone() const = 0; 
  
};

#endif
