#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"


template <unsigned int N>
VertexTrack<N>::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, 
			 float weight) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(false), covAvailable(false), smoothedChi2_(-1.) {}


template <unsigned int N>
VertexTrack<N>::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, float weight,
			 const RefCountedRefittedTrackState & refittedState,
			 float smoothedChi2)
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(true), covAvailable(false), theRefittedState(refittedState),
    smoothedChi2_(smoothedChi2) {}


template <unsigned int N>
VertexTrack<N>::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, float weight, 
			 const RefCountedRefittedTrackState & refittedState,
			 float smoothedChi2, const AlgebraicMatrix3M & tVCov) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(true), covAvailable(true), 
    theRefittedState(refittedState), tkTVCovariance(tVCov),
    smoothedChi2_(smoothedChi2) {}


template <unsigned int N>
typename VertexTrack<N>::AlgebraicVectorN VertexTrack<N>::refittedParamFromEquation() const 
{
  return linearizedTrack()->refittedParamFromEquation(theRefittedState);
}

template class VertexTrack<5>;
template class VertexTrack<6>;
