#include "RecoVertex/VertexTools/interface/DummyTrackToTrackCovCalculator.h"


template <unsigned int N>
typename CachingVertex<N>::TrackToTrackMap 
DummyTrackToTrackCovCalculator<N>::operator() (const CachingVertex<N> &) const
{
  return typename CachingVertex<N>::TrackToTrackMap();
}


template <unsigned int N>
DummyTrackToTrackCovCalculator<N> * DummyTrackToTrackCovCalculator<N>::clone() const
{
  return new DummyTrackToTrackCovCalculator(*this);
}

template class DummyTrackToTrackCovCalculator<5>;
template class DummyTrackToTrackCovCalculator<6>;
