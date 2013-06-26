#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"

template <unsigned int N>
CachingVertex<N> DummyVertexSmoother<N>::smooth( const CachingVertex<N> & vertex ) const
{
  return vertex;
}

template <unsigned int N>
DummyVertexSmoother<N> * DummyVertexSmoother<N>::clone() const
{
  return new DummyVertexSmoother ( * this );
}

template class DummyVertexSmoother<5>;
template class DummyVertexSmoother<6>;
