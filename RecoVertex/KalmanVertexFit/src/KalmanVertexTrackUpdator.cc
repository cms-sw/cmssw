#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
//#include "Utilities/GenUtil/interface/ReferenceCountingPointer.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


RefCountedVertexTrack KalmanVertexTrackUpdator::update
	(const CachingVertex & vertex , RefCountedVertexTrack track) const

{
  pair<RefCountedRefittedTrackState, AlgebraicMatrix33> thePair = 
  	trackRefit(vertex.vertexState(), track->linearizedTrack());

  CachingVertex rVert = updator.remove(vertex, track);

  float smoothedChi2 = helper.vertexChi2(rVert, vertex) +
	helper.trackParameterChi2(track->linearizedTrack(), thePair.first);

  return theVTFactory.vertexTrack(track->linearizedTrack(),
  	vertex.vertexState(), thePair.first, smoothedChi2, thePair.second,
	track->weight());
}

pair<RefCountedRefittedTrackState, AlgebraicMatrix33> 
KalmanVertexTrackUpdator::trackRefit(const VertexState & vertex,
	 RefCountedLinearizedTrackState linTrackState) const

{
  //Vertex position 
  GlobalPoint vertexPosition = vertex.position();

  AlgebraicVector3 vertexCoord;
  vertexCoord(0) = vertexPosition.x();
  vertexCoord(1) = vertexPosition.y();
  vertexCoord(2) = vertexPosition.z();
  AlgebraicSymMatrix33 vertexErrorMatrix = vertex.error().matrix_new();

//track information
  AlgebraicMatrix53 a = linTrackState->positionJacobian();
  AlgebraicMatrix53 b = linTrackState->momentumJacobian();

  AlgebraicVector5 trackParameters = 
  	linTrackState->predictedStateParameters();

  AlgebraicSymMatrix55 trackParametersWeight = 
  	linTrackState->predictedStateWeight();

  AlgebraicSymMatrix33 s = ROOT::Math::SimilarityT(b,trackParametersWeight);
  
  int ifail = ! s.Invert();
  if(ifail !=0) throw VertexException
  	("KalmanVertexTrackUpdator::S matrix inversion failed");
   
  AlgebraicVector3 newTrackMomentumP =  s * (ROOT::Math::Transpose(b)) * trackParametersWeight * 
    (trackParameters - linTrackState->constantTerm() - a*vertexCoord);

  AlgebraicMatrix33 refittedPositionMomentumConvariance = 
    -vertexErrorMatrix * (ROOT::Math::Transpose(a)) * trackParametersWeight * b * s;

  AlgebraicSymMatrix33 refittedMomentumConvariance = s +  
     ROOT::Math::SimilarityT(refittedPositionMomentumConvariance, vertex.weight().matrix_new());

  
 // int matrixSize = 3+3; //refittedMomentumConvariance.num_col();
  AlgebraicMatrix66  covMatrix; //(matrixSize, matrixSize);
  covMatrix.Place_at(refittedPositionMomentumConvariance, 0, 3);
  covMatrix.Place_at(ROOT::Math::Transpose(refittedPositionMomentumConvariance), 3, 0);
  covMatrix.Place_at(vertexErrorMatrix, 0, 0);
  covMatrix.Place_at(refittedMomentumConvariance, 3 ,3);
  ROOT::Math::SVector<double, 21> vup = covMatrix.UpperBlock();

  AlgebraicSymMatrix66 covSymMatrix (vup);

  RefCountedRefittedTrackState refittedTrackState = linTrackState->
	createRefittedTrackState(vertexPosition, newTrackMomentumP, covSymMatrix);

  return pair<RefCountedRefittedTrackState, AlgebraicMatrix33>
  		(refittedTrackState, refittedPositionMomentumConvariance);
} 
