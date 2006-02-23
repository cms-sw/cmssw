#ifndef _VertexTrackUpdator_H_
#define _VertexTrackUpdator_H_

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

/** 
 * Pure abstract base class for VertexTrackUpdators. 
 * Improves the track parameters at the vertex using the vertex constraint. 
 */

class VertexTrackUpdator {

public:

  /**
   * Computes the constrained track parameters
   */
  virtual 
  RefCountedVertexTrack update(const CachingVertex & v, 
			       RefCountedVertexTrack t) const = 0;

  virtual VertexTrackUpdator * clone() const = 0;

};

#endif
