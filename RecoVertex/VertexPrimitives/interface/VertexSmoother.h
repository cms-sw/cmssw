#ifndef _VertexSmoother_H_
#define _VertexSmoother_H_


#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"


/**
 * Pure abstract base class for vertex smoothers 
 * Vertex smoothing includes track refit with vertex constraint, 
 * better estimation of vertex chi-squared etc. 
 */

template <unsigned int N>
class VertexSmoother {

public:

  VertexSmoother() {}
  virtual ~VertexSmoother() {}
  
  /**
   *  Smoothing method
   */
  virtual CachingVertex<N> smooth(const CachingVertex<N> & vertex) const = 0;

  /**
   * Clone method 
   */
  virtual VertexSmoother * clone() const = 0; 

};

#endif
