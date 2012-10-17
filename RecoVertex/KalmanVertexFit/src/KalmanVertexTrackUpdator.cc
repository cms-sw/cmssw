#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
//#include "Utilities/GenUtil/interface/ReferenceCountingPointer.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"

#include<iostream>


template <unsigned int N>
typename CachingVertex<N>::RefCountedVertexTrack
KalmanVertexTrackUpdator<N>::update
	(const CachingVertex<N> & vertex , RefCountedVertexTrack track) const

{
  trackMatrixPair thePair = 
  	trackRefit(vertex.vertexState(), track->linearizedTrack(), track->weight() );

  VertexState rVert = updator.positionUpdate (vertex.vertexState(), track->linearizedTrack(),
			track->weight(), -1);

  std::pair<bool, double> result = helper.trackParameterChi2(track->linearizedTrack(), thePair.first);
  float smoothedChi2 = helper.vertexChi2(rVert, vertex.vertexState()) + result.second;

  return theVTFactory.vertexTrack(track->linearizedTrack(),
  	vertex.vertexState(), thePair.first, smoothedChi2, thePair.second,
	track->weight());
}

template <unsigned int N>
typename KalmanVertexTrackUpdator<N>::trackMatrixPair
KalmanVertexTrackUpdator<N>::trackRefit(const VertexState & vertex,
	 KalmanVertexTrackUpdator<N>::RefCountedLinearizedTrackState linTrackState,
	 float weight) const
{
  typedef ROOT::Math::SVector<double,N> AlgebraicVectorN;
  typedef ROOT::Math::SVector<double,N-2> AlgebraicVectorM;
  typedef ROOT::Math::SMatrix<double,N,3,ROOT::Math::MatRepStd<double,N,3> > AlgebraicMatrixN3;
  typedef ROOT::Math::SMatrix<double,3,N,ROOT::Math::MatRepStd<double,3,N> > AlgebraicMatrix3N;
  typedef ROOT::Math::SMatrix<double,3,N-2,ROOT::Math::MatRepStd<double,3,N-2> > AlgebraicMatrix3M;
  typedef ROOT::Math::SMatrix<double,N,N-2,ROOT::Math::MatRepStd<double,N,N-2> > AlgebraicMatrixNM;
  typedef ROOT::Math::SMatrix<double,N-2,3,ROOT::Math::MatRepStd<double,N-2,3> > AlgebraicMatrixM3;
  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > AlgebraicSymMatrixNN;
//   typedef ROOT::Math::SMatrix<double,N+1,N+1,ROOT::Math::MatRepSym<double,N+1> > AlgebraicSymMatrixOO;
  typedef ROOT::Math::SMatrix<double,N+1,N+1,ROOT::Math::MatRepStd<double,N+1,N+1> > AlgebraicMatrixOO;
  typedef ROOT::Math::SMatrix<double,N-2,N-2,ROOT::Math::MatRepSym<double,N-2> > AlgebraicSymMatrixMM;

  //Vertex position 
  GlobalPoint vertexPosition = vertex.position();

  AlgebraicVector3 vertexCoord;
  vertexCoord(0) = vertexPosition.x();
  vertexCoord(1) = vertexPosition.y();
  vertexCoord(2) = vertexPosition.z();
  const AlgebraicSymMatrix33 vertexErrorMatrix = vertex.error().matrix_new();

//track information
  const AlgebraicMatrixN3 a = linTrackState->positionJacobian();
  const AlgebraicMatrixNM b = linTrackState->momentumJacobian();

//   AlgebraicVectorN trackParameters = 
//   	linTrackState->predictedStateParameters();

  int ifail;
  AlgebraicSymMatrixNN trackParametersWeight = 
  	linTrackState->predictedStateWeight(ifail);

  AlgebraicSymMatrixMM s = ROOT::Math::SimilarityT(b,trackParametersWeight);
  
  if (!invertPosDefMatrix(s)) 
      throw VertexException
  	("KalmanVertexTrackUpdator::S matrix inversion failed");
   
  //                                    NN           NM  MM
  AlgebraicMatrixNM twbs =  trackParametersWeight *  b * s;


  AlgebraicVectorN vv = 
    linTrackState->predictedStateParameters() - linTrackState->constantTerm() - a*vertexCoord;
  //                                   MM                MN                    NN
  //  AlgebraicVectorM newTrackMomentumP =  s * (ROOT::Math::Transpose(b)) * trackParametersWeight * vv;
  AlgebraicVectorM newTrackMomentumP = ROOT::Math::Transpose(twbs) * vv;

   //AlgebraicMatrix3M refittedPositionMomentumConvariance = 
  //        33                        3N                    NN                  NM  MM
  //  -vertexErrorMatrix * (ROOT::Math::Transpose(a)) * trackParametersWeight * b * s;

  AlgebraicMatrix3N tmpM1 = -vertexErrorMatrix * (ROOT::Math::Transpose(a));
  AlgebraicMatrix3M  refittedPositionMomentumConvariance  = tmpM1 * twbs;

  AlgebraicSymMatrixMM refittedMomentumConvariance = s/weight +  
     ROOT::Math::SimilarityT(refittedPositionMomentumConvariance, vertex.weight().matrix_new());

  
 // int matrixSize = 3+3; //refittedMomentumConvariance.num_col();
  AlgebraicMatrixOO  covMatrix; //(matrixSize, matrixSize);
  covMatrix.Place_at(refittedPositionMomentumConvariance, 0, 3);
  covMatrix.Place_at(ROOT::Math::Transpose(refittedPositionMomentumConvariance), 3, 0);
  covMatrix.Place_at(vertexErrorMatrix, 0, 0);
  covMatrix.Place_at(refittedMomentumConvariance, 3 ,3);

  AlgebraicSymMatrixOO covSymMatrix(covMatrix.LowerBlock());

  RefCountedRefittedTrackState refittedTrackState = linTrackState->
	createRefittedTrackState(vertexPosition, newTrackMomentumP, covSymMatrix);

  return trackMatrixPair(refittedTrackState, covSymMatrix);
//   		(refittedTrackState, refittedPositionMomentumConvariance);
} 

template class KalmanVertexTrackUpdator<5>;
template class KalmanVertexTrackUpdator<6>;
