#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"

CachingVertex DummyVertexSmoother::smooth( const CachingVertex & vertex ) const
{
  return vertex;
};

DummyVertexSmoother * DummyVertexSmoother::clone() const
{
  return new DummyVertexSmoother ( * this );
};
