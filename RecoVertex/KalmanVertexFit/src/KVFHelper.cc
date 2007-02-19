#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"


double KVFHelper::vertexChi2(const CachingVertex vertexA, 
	const CachingVertex vertexB) const
{
  return vertexChi2(vertexA.vertexState(), vertexB.vertexState());
};


double KVFHelper::vertexChi2(const VertexState vertexA,
	const VertexState vertexB) const
{
  double vetexChi2;
  GlobalPoint inPosition = vertexA.position();
  GlobalPoint fnPosition = vertexB.position();
  GlobalError inError = vertexA.error();
  AlgebraicVector pDiff(3);
  pDiff[1] =  inPosition.x() - fnPosition.x();
  pDiff[2] =  inPosition.y() - fnPosition.y();
  pDiff[3] =  inPosition.z() - fnPosition.z();
  int ifail;
  vetexChi2 = inError.matrix().inverse(ifail).similarity(pDiff);
  if (ifail!=0) vetexChi2 = 0;
  return vetexChi2;
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

