#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"
using namespace std;

template <unsigned int N>
double KVFHelper<N>::vertexChi2(const CachingVertex<N> & vertexA, 
	const CachingVertex<N> & vertexB) const
{
  return vertexChi2(vertexA.vertexState(), vertexB.vertexState());
}


template <unsigned int N>
double KVFHelper<N>::vertexChi2(const VertexState & vertexA,
	const VertexState & vertexB) const
{
// std::cout <<"Start\n";
  GlobalPoint inPosition = vertexA.position();
  GlobalPoint fnPosition = vertexB.position();
//   std::cout << inPosition<< fnPosition<<std::endl;

  AlgebraicVector3 oldVertexPositionV;
  oldVertexPositionV(0) = inPosition.x();
  oldVertexPositionV(1) = inPosition.y();
  oldVertexPositionV(2) = inPosition.z();

  AlgebraicVector3 newVertexPositionV;
  newVertexPositionV(0) = fnPosition.x();
  newVertexPositionV(1) = fnPosition.y();
  newVertexPositionV(2) = fnPosition.z();


  AlgebraicVector3 positionResidual = newVertexPositionV - oldVertexPositionV;

  return ROOT::Math::Similarity(positionResidual, vertexA.weight().matrix_new());
}


template <unsigned int N>
typename KVFHelper<N>::BDpair KVFHelper<N>::trackParameterChi2(const RefCountedVertexTrack track) const
{
  return trackParameterChi2(track->linearizedTrack(), track->refittedState());
}


template <unsigned int N>
typename KVFHelper<N>::BDpair KVFHelper<N>::trackParameterChi2(
	const RefCountedLinearizedTrackState linTrack,
	const RefCountedRefittedTrackState refittedTrackState) const
{

  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > AlgebraicSymMatrixNN;
  typedef ROOT::Math::SVector<double,N> AlgebraicVectorN;

  AlgebraicVectorN parameterResiduals = linTrack->predictedStateParameters() -
  	linTrack->refittedParamFromEquation(refittedTrackState);
  linTrack->checkParameters(parameterResiduals);
  int error;
  float lChi2 = ROOT::Math::Similarity(parameterResiduals, linTrack->predictedStateWeight(error));
  if (error != 0) return BDpair(false, -1.);
  return BDpair(true, lChi2);
}

template class KVFHelper<5>;
template class KVFHelper<6>;
