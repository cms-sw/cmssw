#ifndef _DummyVertexSmoother_H_
#define _DummyVertexSmoother_H_

#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"

/**
 *  A dummy vertex smoother. Input = Output.
 */

template <unsigned int N>
class DummyVertexSmoother : public VertexSmoother<N> {
public:
  CachingVertex<N> smooth(const CachingVertex<N> & ) const override;
  DummyVertexSmoother * clone() const override;
};

#endif
