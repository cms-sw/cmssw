#include "RecoVertex/KalmanVertexFit/interface/KalmanSmoothedVertexChi2Estimator.h"


float KalmanSmoothedVertexChi2Estimator::estimate(const CachingVertex & vertex) const
{
//initial vertex part
  float v_part = 0.;
  float returnChi = 0.;
  
  if (vertex.hasPrior()) {
    v_part = priorVertexChi2(vertex.priorVertexState(), vertex.vertexState());
  }
 
//vector of tracks part
  vector<RefCountedVertexTrack> tracks = vertex.tracks();
  float sum = 0.;
  for(vector<RefCountedVertexTrack>::iterator i = tracks.begin(); i != tracks.end(); i++)
  {
   sum += trackParameterChi2((*i)->linearizedTrack(), (*i)->refittedState());
  }
 returnChi = v_part + sum;
 return returnChi;   
}

double KalmanSmoothedVertexChi2Estimator::priorVertexChi2(
	const VertexState priorVertex, const VertexState fittedVertex) const
{
  double vetexChi2;
  GlobalPoint inPosition = priorVertex.position();
  GlobalPoint fnPosition = fittedVertex.position();
  GlobalError inError = priorVertex.error();
  AlgebraicVector pDiff(3);
  pDiff[1] =  inPosition.x() - fnPosition.x();
  pDiff[2] =  inPosition.y() - fnPosition.y();
  pDiff[3] =  inPosition.z() - fnPosition.z();
  int ifail;
  vetexChi2 = inError.matrix().inverse(ifail).similarity(pDiff);
  if (ifail!=0) vetexChi2 = 0;
  return vetexChi2;
}

float KalmanSmoothedVertexChi2Estimator::trackParameterChi2(
	const RefCountedLinearizedTrackState linTrack,
	const RefCountedRefittedTrackState refittedTrackState) const
{
  AlgebraicVector parameterResiduals = linTrack->predictedStateParameters() -
  	linTrack->refittedParamFromEquation(refittedTrackState);
  AlgebraicSymMatrix trackParametersWeight = linTrack->predictedStateWeight();

  float lChi2 =trackParametersWeight.similarity(parameterResiduals);
  return (lChi2);
}

