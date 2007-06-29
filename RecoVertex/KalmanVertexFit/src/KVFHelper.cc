#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"
using namespace std;

double KVFHelper::vertexChi2(const CachingVertex & vertexA, 
	const CachingVertex & vertexB) const
{
  return vertexChi2(vertexA.vertexState(), vertexB.vertexState());
}


double KVFHelper::vertexChi2(const VertexState & vertexA,
	const VertexState & vertexB) const
{
// cout <<"Start\n";
  GlobalPoint inPosition = vertexA.position();
  GlobalPoint fnPosition = vertexB.position();
//   cout << inPosition<< fnPosition<<endl;

  AlgebraicVector oldVertexPositionV(3);
  oldVertexPositionV[0] = inPosition.x();
  oldVertexPositionV[1] = inPosition.y();
  oldVertexPositionV[2] = inPosition.z();

  AlgebraicVector newVertexPositionV(3);
  newVertexPositionV[0] = fnPosition.x();
  newVertexPositionV[1] = fnPosition.y();
  newVertexPositionV[2] = fnPosition.z();

  AlgebraicVector positionResidual = newVertexPositionV - oldVertexPositionV;

  return vertexA.weight().matrix().similarity(positionResidual);
}


float KVFHelper::trackParameterChi2(const RefCountedVertexTrack track) const
{
  return trackParameterChi2(track->linearizedTrack(), track->refittedState());
}


float KVFHelper::trackParameterChi2(
	const RefCountedLinearizedTrackState linTrack,
	const RefCountedRefittedTrackState refittedTrackState) const
{
  AlgebraicVector parameterResiduals = linTrack->predictedStateParameters() -
  	linTrack->refittedParamFromEquation(refittedTrackState);
  AlgebraicSymMatrix trackParametersWeight = linTrack->predictedStateWeight();

  float lChi2 =trackParametersWeight.similarity(parameterResiduals);
  return (lChi2);
}

