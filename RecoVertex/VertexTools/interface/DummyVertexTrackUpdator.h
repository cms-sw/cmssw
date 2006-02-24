#ifndef _DummyVertexTrackUpdator_H_
#define _DummyVertexTrackUpdator_H_

#include "RecoVertex/VertexPrimitives/interface/VertexTrackUpdator.h"

/** 
 * \class DummyVertexTrackUpdator
 * Returns RefCountedVertexTrack unchanged
 */

class DummyVertexTrackUpdator : public VertexTrackUpdator {

public:

  /**
   * Computes the constrained track parameters
   */
  virtual 
  RefCountedVertexTrack update(const CachingVertex & v, 
			       RefCountedVertexTrack t) const;
  virtual DummyVertexTrackUpdator * clone() const;

};

#endif
