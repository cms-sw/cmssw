#ifndef _DummyVertexSmoother_H_
#define _DummyVertexSmoother_H_

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"

/**
 *  A dummy vertex smoother. Input = Output.
 */

class DummyVertexSmoother : public VertexSmoother {
public:
  CachingVertex smooth(const CachingVertex & ) const;
  DummyVertexSmoother * clone() const;
};

#endif
