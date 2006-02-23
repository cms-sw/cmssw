#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"


VertexTrack::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, 
			 float weight) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(false), covAvailable(false) {}


VertexTrack::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, 
			 float weight, const RefCountedRefittedTrackState & refittedState) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(true), covAvailable(false), theRefittedState(refittedState) {}


VertexTrack::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, 
			 float weight, const RefCountedRefittedTrackState & refittedState,
			 const AlgebraicMatrix & tVCov) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(true), covAvailable(true), 
    theRefittedState(refittedState), tkTVCovariance(tVCov) {}


AlgebraicVector VertexTrack::refittedParamFromEquation() const 
{
  AlgebraicVector vertexPosition(3);
  vertexPosition[0] = theRefittedState->position().x();
  vertexPosition[1] = theRefittedState->position().y();
  vertexPosition[2] = theRefittedState->position().z();
  AlgebraicVector momentumAtVertex = theRefittedState->momentumVector();

  AlgebraicMatrix thePositionJacobian = linearizedTrack()->positionJacobian();
  AlgebraicMatrix theMomentumJacobian = linearizedTrack()->momentumJacobian();
  AlgebraicVector theResidual = linearizedTrack()->constantTerm();
  
  AlgebraicVector rtp = ( theResidual + 
			  thePositionJacobian * vertexPosition +
			  theMomentumJacobian * momentumAtVertex);
  
  return rtp;
}
