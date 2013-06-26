#include "RecoVertex/VertexTools/interface/DummyVertexTrackUpdator.h"


template <unsigned int N>
typename CachingVertex<N>::RefCountedVertexTrack 
DummyVertexTrackUpdator<N>::update(const CachingVertex<N> & v, 
  typename CachingVertex<N>::RefCountedVertexTrack t) const
{
  return t;
}


template <unsigned int N>
DummyVertexTrackUpdator<N> * DummyVertexTrackUpdator<N>::clone() const
{
  return new DummyVertexTrackUpdator(*this);
}

template class DummyVertexTrackUpdator<5>;
template class DummyVertexTrackUpdator<6>;
