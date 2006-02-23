#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
// #include "Vertex/VertexPrimitives/interface/PerigeeRefittedTrackState.h"
// #include "CommonReco/PatternTools/interface/PerigeeConversions.h"

AlgebraicVector 
LinearizedTrackState::refittedParamFromEquation(
	const RefCountedRefittedTrackState & theRefittedState) const 
{
  AlgebraicVector vertexPosition(3);
  vertexPosition[0] = theRefittedState->position().x();
  vertexPosition[1] = theRefittedState->position().y();
  vertexPosition[2] = theRefittedState->position().z();

  AlgebraicVector rtp = ( constantTerm() + 
		       positionJacobian() * vertexPosition +
		       momentumJacobian() * theRefittedState->momentumVector());
  
  return rtp;
}
