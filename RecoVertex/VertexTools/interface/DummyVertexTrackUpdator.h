#ifndef _DummyVertexTrackUpdator_H_
#define _DummyVertexTrackUpdator_H_

#include "RecoVertex/VertexPrimitives/interface/VertexTrackUpdator.h"

/** 
 * \class DummyVertexTrackUpdator
 * Returns RefCountedVertexTrack unchanged
 */

template <unsigned int N>
class DummyVertexTrackUpdator : public VertexTrackUpdator<N> {

public:

  /**
   * Computes the constrained track parameters
   */
  typename CachingVertex<N>::RefCountedVertexTrack
	update(const CachingVertex<N> & v, 
	typename CachingVertex<N>::RefCountedVertexTrack t) const override;
  DummyVertexTrackUpdator * clone() const override;

};

#endif
