#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"


VertexTrack::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, 
			 float weight) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(false), covAvailable(false), smoothedChi2_(-1.) {}


VertexTrack::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, float weight,
			 const RefCountedRefittedTrackState & refittedState,
			 float smoothedChi2)
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(true), covAvailable(false), theRefittedState(refittedState),
    smoothedChi2_(smoothedChi2) {}


VertexTrack::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, float weight, 
			 const RefCountedRefittedTrackState & refittedState,
			 float smoothedChi2, const AlgebraicMatrix & tVCov) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(true), covAvailable(true), 
    theRefittedState(refittedState), tkTVCovariance(tVCov),
    smoothedChi2_(smoothedChi2) {}


AlgebraicVector VertexTrack::refittedParamFromEquation() const 
{
  return linearizedTrack()->refittedParamFromEquation(theRefittedState);
}
