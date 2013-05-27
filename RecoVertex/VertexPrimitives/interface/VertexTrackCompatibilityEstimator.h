#ifndef _VertexTrackCompatibilityEstimator_H
#define _VertexTrackCompatibilityEstimator_H

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include <climits>

/**
 * Pure abstract base class for compatibility estimators 
 * (distance, chi-squared, etc.)
 */

template <unsigned int N>
class VertexTrackCompatibilityEstimator {
 
public:

  typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef typename VertexTrack<N>::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;
  typedef typename std::pair <bool, double> BDpair;

  VertexTrackCompatibilityEstimator(){}
  virtual ~VertexTrackCompatibilityEstimator(){}
  
  /**
   * Methods giving back the compatibility estimation
   */
  virtual BDpair estimate(const CachingVertex<N> & v, 
			  const RefCountedLinearizedTrackState track,
			  unsigned int hint=UINT_MAX) const = 0;

  virtual BDpair estimate(const reco::Vertex & v, 
			 const reco::TransientTrack & track) const = 0;

  // obsolete ?
  virtual BDpair estimate(const CachingVertex<N> & v, 
			  const RefCountedVertexTrack track, unsigned int hint=UINT_MAX) const = 0;
  /**
   * Clone method 
   */
  virtual VertexTrackCompatibilityEstimator<N> * clone() const = 0; 
  
};

#endif
