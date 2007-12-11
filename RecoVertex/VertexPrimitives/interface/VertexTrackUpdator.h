#ifndef _VertexTrackUpdator_H_
#define _VertexTrackUpdator_H_

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

/** 
 * Pure abstract base class for VertexTrackUpdators. 
 * Improves the track parameters at the vertex using the vertex constraint. 
 */

template <unsigned int N>
class VertexTrackUpdator {

public:

  /**
   * Computes the constrained track parameters
   */
  virtual typename CachingVertex<N>::RefCountedVertexTrack 
	update(const CachingVertex<N> & v, 
	typename CachingVertex<N>::RefCountedVertexTrack t) const = 0;

  virtual VertexTrackUpdator * clone() const = 0;
  virtual ~VertexTrackUpdator() {};

};

#endif
