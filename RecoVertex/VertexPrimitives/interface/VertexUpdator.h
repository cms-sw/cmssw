#ifndef VertexUpdator_H
#define VertexUpdator_H

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

/** 
 * Pure abstract base class for VertexUpdators. 
 * Updates CachingVertex with one VertexTrack. 
 */

template <unsigned int N>
class VertexUpdator {

public:
  
  typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;

  /**
   * Default Constructor
   */  
   VertexUpdator() {}
    
   virtual ~VertexUpdator() {}
    
  /**
   * Method updating the vertex, with the information contained 
   * in the track.
   */
  virtual CachingVertex<N> add(const CachingVertex<N> & v,
	const typename CachingVertex<N>::RefCountedVertexTrack  t) const = 0;

  virtual CachingVertex<N> remove(const CachingVertex<N> & v,
	const typename CachingVertex<N>::RefCountedVertexTrack  t) const = 0;

  virtual VertexUpdator * clone() const = 0;  

};


#endif
